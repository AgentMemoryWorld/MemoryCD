"""
Shared evaluation core for MemoryCD.

This module contains all the components common to single-domain and cross-domain
evaluation:

- MetadataLoader: multi-domain item-metadata loader with pickle cache
- EvaluationLogger: per-prediction JSONL + summary JSON writer
- *Evaluator classes: rating, text generation, and item ranking metrics
- LLMPredictor: prompt builders + LLM calls + response parsing for all 4 tasks
- build_method(): factory for LongContext / RAG memory selectors
- load_cross_domain_jsonl(): cross-domain loader (target + source domains)
- load_single_domain_from_cross_domain(): single-domain loader (one domain only)

Both evaluation entry-point scripts (single and cross domain) import everything
they need from here, so there is no duplicated logic between them.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import pickle
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api import get_active_provider, get_default_model, get_gateway_client
from methods import BaseMethod, LongContextMethod, RAGMethod

# Optional metric dependencies -------------------------------------------------

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. Install with: pip install rouge-score", file=sys.stderr)

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not available. Install with: pip install bert-score", file=sys.stderr)

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    BLEU_AVAILABLE = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk not available for BLEU score. Install with: pip install nltk", file=sys.stderr)


# Fixed cross-domain target list. Keep in sync with the data-prep pipeline that
# produces the cross-domain JSONL.
DOMAINS: List[str] = [
    "Beauty_and_Personal_Care",
    "Books",
    "Electronics",
    "Home_and_Kitchen",
]


# ----------------------------------------------------------------------------- #
# File I/O
# ----------------------------------------------------------------------------- #

def open_text_auto(path: Path) -> io.TextIOBase:
    """Open a text file, transparently handling .gz."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(
            gzip.open(path, mode="rb"), encoding="utf-8", errors="replace"
        )
    return path.open("r", encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------------- #
# Metadata loader (multi-domain, cached)
# ----------------------------------------------------------------------------- #

class MetadataLoader:
    """Loads and caches Amazon item metadata for any subset of the four domains."""

    def __init__(
        self,
        meta_dir: Path,
        domains: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.meta_dir = meta_dir
        self.domains = list(domains) if domains else list(DOMAINS)
        self.use_cache = use_cache

        if cache_dir is None:
            cache_dir = meta_dir / "cache_metadata"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_key = "_".join(sorted(self.domains))
        self.cache_file = self.cache_dir / f"metadata_{cache_key}.pkl"

        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._load_all_metadata()

    def _load_from_cache(self) -> bool:
        if not self.use_cache or not self.cache_file.exists():
            return False
        try:
            print(f"Loading cached metadata from: {self.cache_file}", file=sys.stderr)
            with open(self.cache_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"  Loaded {len(self.metadata):,} cached metadata entries", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Warning: failed to load metadata cache: {e}", file=sys.stderr)
            return False

    def _save_to_cache(self) -> None:
        if not self.use_cache:
            return
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved metadata cache to: {self.cache_file}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: failed to save metadata cache: {e}", file=sys.stderr)

    def _load_all_metadata(self) -> None:
        if self._load_from_cache():
            return

        print(f"Loading metadata for domains {self.domains} from: {self.meta_dir}", file=sys.stderr)
        for domain in self.domains:
            meta_path = self.meta_dir / f"meta_{domain}.jsonl.gz"
            if not meta_path.exists():
                meta_path = self.meta_dir / f"meta_{domain}.jsonl"
            if not meta_path.exists():
                print(f"Warning: metadata file not found: {meta_path}", file=sys.stderr)
                continue

            count = 0
            try:
                with open_text_auto(meta_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        parent_asin = item.get("parent_asin")
                        if parent_asin:
                            self.metadata[parent_asin] = item
                            count += 1
                print(f"  {domain}: {count:,} items", file=sys.stderr)
            except Exception as e:
                print(f"  Error loading {domain}: {e}", file=sys.stderr)

        print(f"Total metadata entries: {len(self.metadata):,}", file=sys.stderr)
        self._save_to_cache()

    def get_metadata(self, parent_asin: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get(parent_asin)


# ----------------------------------------------------------------------------- #
# Evaluation logger
# ----------------------------------------------------------------------------- #

class EvaluationLogger:
    """Writes a per-prediction JSONL and a final summary JSON."""

    def __init__(
        self,
        log_dir: Path,
        run_id: str,
    ):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

        self.predictions_log_path = log_dir / f"predictions_{run_id}.jsonl"
        self.summary_log_path = log_dir / f"summary_{run_id}.json"
        self.predictions_file = open(self.predictions_log_path, "w", encoding="utf-8")
        self.prediction_count = 0

    def log_prediction(
        self,
        user_id: str,
        memory: List[Dict[str, Any]],
        target_item: Dict[str, Any],
        prompt: str,
        llm_response: str,
        prediction: Any,
        ground_truth: Any,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        entry = {
            "prediction_id": self.prediction_count,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "num_memory_items": len(memory),
            "memory": memory,
            "target_item": target_item,
            "prompt": prompt,
            "llm_response": llm_response,
            "prediction": prediction,
            "ground_truth": ground_truth,
        }
        if metrics:
            entry["metrics"] = metrics
        self.predictions_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.predictions_file.flush()
        self.prediction_count += 1

    def log_summary(self, results: Dict[str, Any]) -> None:
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "log_files": {
                "predictions": str(self.predictions_log_path),
                "summary": str(self.summary_log_path),
            },
        }
        with open(self.summary_log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def close(self) -> None:
        if hasattr(self, "predictions_file"):
            self.predictions_file.close()


def build_run_id(
    setting: str,
    task_name: str,
    method_name: str,
    model_name: str,
    domain_part: str,
    max_users: Optional[int],
    max_memory_items: Optional[int],
    num_test: int,
) -> str:
    """Build a deterministic, descriptive run ID for log filenames."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_name.replace("/", "_")
    users_str = f"users{max_users}" if max_users is not None else "usersAll"
    mem_str = f"mem{max_memory_items}" if max_memory_items is not None else "memAll"
    test_str = f"test{num_test}"
    return (
        f"{setting}_{domain_part}_{task_name}_{method_name}_{model_safe}"
        f"_{users_str}_{mem_str}_{test_str}_{timestamp}"
    )


# ----------------------------------------------------------------------------- #
# Memory / test split helpers
# ----------------------------------------------------------------------------- #

def split_last_n(interactions: List[Dict[str, Any]], num_test: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (memory, test) by sorting on timestamp and taking the last `num_test` for test."""
    if not interactions:
        return [], []
    sorted_inter = sorted(interactions, key=lambda x: x.get("timestamp", 0))
    n = len(sorted_inter)
    if n <= num_test:
        if n == 1:
            return [], sorted_inter
        return sorted_inter[:-1], sorted_inter[-1:]
    return sorted_inter[: n - num_test], sorted_inter[n - num_test :]


# ----------------------------------------------------------------------------- #
# Evaluators
# ----------------------------------------------------------------------------- #

class RatingPredictionEvaluator:
    """MAE / RMSE for rating prediction."""

    def __init__(self) -> None:
        self.predictions: List[float] = []
        self.ground_truth: List[float] = []

    def add_prediction(self, predicted: float, actual: float) -> None:
        self.predictions.append(predicted)
        self.ground_truth.append(actual)

    def compute_metrics(self) -> Dict[str, float]:
        if not self.predictions:
            return {"mae": 0.0, "rmse": 0.0, "count": 0}
        n = len(self.predictions)
        mae = sum(abs(p - t) for p, t in zip(self.predictions, self.ground_truth)) / n
        mse = sum((p - t) ** 2 for p, t in zip(self.predictions, self.ground_truth)) / n
        return {"mae": mae, "rmse": math.sqrt(mse), "count": n}


class TextGenerationEvaluator:
    """ROUGE-1/L, BLEU-1/4, BERTScore for review summarization & generation."""

    def __init__(self) -> None:
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True) if ROUGE_AVAILABLE else None
        self.rouge1_scores: List[float] = []
        self.rougeL_scores: List[float] = []
        self.bleu1_scores: List[float] = []
        self.bleu4_scores: List[float] = []
        self.predictions_list: List[str] = []
        self.references_list: List[str] = []

    def add_prediction(self, predicted: str, reference: str) -> Dict[str, float]:
        self.predictions_list.append(predicted)
        self.references_list.append(reference)
        result: Dict[str, float] = {}

        if self.scorer:
            scores = self.scorer.score(reference, predicted)
            r1, rl = scores["rouge1"].fmeasure, scores["rougeL"].fmeasure
            self.rouge1_scores.append(r1)
            self.rougeL_scores.append(rl)
            result["rouge1_f"] = r1
            result["rougeL_f"] = rl
        else:
            result["rouge1_f"] = 0.0
            result["rougeL_f"] = 0.0

        if BLEU_AVAILABLE:
            try:
                ref_tokens = reference.split()
                pred_tokens = predicted.split()
                smoothing = SmoothingFunction()
                b1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing.method1)
                b4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1)
                self.bleu1_scores.append(b1)
                self.bleu4_scores.append(b4)
                result["bleu1"] = b1
                result["bleu4"] = b4
            except Exception as e:
                print(f"Warning: BLEU computation failed: {e}", file=sys.stderr)
                self.bleu1_scores.append(0.0)
                self.bleu4_scores.append(0.0)
                result["bleu1"] = 0.0
                result["bleu4"] = 0.0
        else:
            result["bleu1"] = 0.0
            result["bleu4"] = 0.0

        return result

    def compute_metrics(self) -> Dict[str, float]:
        if not self.rouge1_scores:
            return {
                "rouge1": 0.0, "rougeL": 0.0, "bleu1": 0.0, "bleu4": 0.0,
                "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0,
                "count": 0,
            }

        n = len(self.rouge1_scores)
        avg_r1 = sum(self.rouge1_scores) / n
        avg_rl = sum(self.rougeL_scores) / n
        avg_b1 = sum(self.bleu1_scores) / len(self.bleu1_scores) if self.bleu1_scores else 0.0
        avg_b4 = sum(self.bleu4_scores) / len(self.bleu4_scores) if self.bleu4_scores else 0.0

        bs_p = bs_r = bs_f = 0.0
        if BERTSCORE_AVAILABLE and self.predictions_list:
            try:
                print("Computing BERTScore (this may take a moment)...", file=sys.stderr)
                P, R, F1 = bert_score(self.predictions_list, self.references_list, lang="en", verbose=False, device="cpu")
                bs_p, bs_r, bs_f = P.mean().item(), R.mean().item(), F1.mean().item()
                print(f"BERTScore: P={bs_p:.4f}, R={bs_r:.4f}, F1={bs_f:.4f}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: BERTScore computation failed: {e}", file=sys.stderr)

        return {
            "rouge1": avg_r1, "rougeL": avg_rl, "bleu1": avg_b1, "bleu4": avg_b4,
            "bertscore_precision": bs_p, "bertscore_recall": bs_r, "bertscore_f1": bs_f,
            "count": n,
        }


class ItemRankingEvaluator:
    """NDCG@K and Recall@K for single-relevant-item ranking."""

    def __init__(self, k_values: Optional[List[int]] = None):
        self.k_values = list(k_values) if k_values else [1, 3, 5]
        self.ranked_lists: List[List[str]] = []
        self.ground_truth_asins: List[str] = []

    def add_prediction(self, ranked_asins: List[str], ground_truth_asin: str) -> None:
        self.ranked_lists.append(ranked_asins)
        self.ground_truth_asins.append(ground_truth_asin)

    @staticmethod
    def _ndcg_at_k(ranked: List[str], gt: str, k: int) -> float:
        if not ranked or k <= 0:
            return 0.0
        topk = ranked[:k]
        rel = [1.0 if a == gt else 0.0 for a in topk]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
        idcg = 1.0  # 1/log2(2) with a single relevant item at rank 1
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _recall_at_k(ranked: List[str], gt: str, k: int) -> float:
        if not ranked or k <= 0:
            return 0.0
        return 1.0 if gt in ranked[:k] else 0.0

    def compute_metrics(self) -> Dict[str, float]:
        if not self.ranked_lists:
            result: Dict[str, float] = {"count": 0}
            for k in self.k_values:
                result[f"ndcg@{k}"] = 0.0
                result[f"recall@{k}"] = 0.0
            return result

        n = len(self.ranked_lists)
        result = {"count": n}
        for k in self.k_values:
            ndcgs = [self._ndcg_at_k(r, g, k) for r, g in zip(self.ranked_lists, self.ground_truth_asins)]
            recalls = [self._recall_at_k(r, g, k) for r, g in zip(self.ranked_lists, self.ground_truth_asins)]
            result[f"ndcg@{k}"] = sum(ndcgs) / n
            result[f"recall@{k}"] = sum(recalls) / n
        return result


# ----------------------------------------------------------------------------- #
# LLM predictor (prompts + API calls + parsing) for all 4 tasks
# ----------------------------------------------------------------------------- #

class LLMPredictor:
    """Unified LLM-based predictor for the 4 evaluation tasks."""

    def __init__(
        self,
        model: Optional[str] = None,
        method: Optional[BaseMethod] = None,
        include_metadata: bool = True,
    ):
        self.client = get_gateway_client()
        self.model = model or get_default_model(get_active_provider())
        self.method = method
        self.include_metadata = include_metadata

    # ---- helpers ----

    def _format_interaction(
        self,
        interaction: Dict[str, Any],
        include_meta: bool = True,
        include_rating: bool = True,
        include_review: bool = True,
    ) -> str:
        parts: List[str] = []
        if "domain_category" in interaction:
            parts.append(f"Domain: {interaction['domain_category']}")
        if "title" in interaction:
            parts.append(f"Title: {interaction['title']}")
        if include_rating and "rating" in interaction:
            parts.append(f"Rating: {interaction['rating']}/5")
        if include_review and "text" in interaction:
            text = interaction["text"] or ""
            if len(text) > 200:
                text = text[:200] + "..."
            parts.append(f"Review: {text}")
        if include_meta and "item_meta" in interaction:
            meta = interaction["item_meta"]
            if "title" in meta and meta["title"] != interaction.get("title"):
                parts.append(f"Product: {meta['title']}")
            if "average_rating" in meta:
                parts.append(f"Product Avg Rating: {meta['average_rating']}")
            if "main_category" in meta:
                parts.append(f"Category: {meta['main_category']}")
        return "\n  ".join(parts)

    def _select_memory(
        self,
        memory: List[Dict[str, Any]],
        target_item: Dict[str, Any],
        task: str,
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if self.method is None:
            return memory
        return self.method.select_memory(memory, target_item, task, user_id=user_id)

    def _call_llm(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"Error in LLM API call: {e}", file=sys.stderr)
            return ""

    # ---- prompt builders ----

    def _build_rating_prediction_prompt(self, memory, target_item, user_id):
        recent = self._select_memory(memory, target_item, "rating_prediction", user_id)
        parts = [
            "You are a personalized rating prediction system. Based on a user's purchase and review history, "
            "predict what rating (1-5) they would give to a new product.",
            "",
            "User's Recent Review History:",
        ]
        for i, inter in enumerate(recent, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        parts.extend([
            "",
            "",
            "Target Product to Rate:",
            self._format_interaction(target_item, self.include_metadata, include_rating=False, include_review=False),
            "",
            "",
            "Based on this user's history and preferences, predict the rating (1-5) they would give to the target product.",
            "Respond with ONLY a single integer between 1 and 5.",
        ])
        return "\n".join(parts)

    def _build_review_summarization_prompt(self, memory, target_item, user_id):
        recent = self._select_memory(memory, target_item, "review_summarization", user_id)
        parts = [
            "You are a personalized review summarization system. Based on a user's review history and writing style, "
            "generate a concise review title that summarizes their full review.",
            "",
            "User's Recent Review History (showing their writing style and preferences):",
        ]
        for i, inter in enumerate(recent, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        rating = target_item.get("rating", "N/A")
        review = target_item.get("text", "")
        parts.extend([
            "",
            "",
            "Target Product and Review to Summarize:",
            self._format_interaction(target_item, self.include_metadata, include_rating=False, include_review=False),
            f"User's Rating: {rating}/5",
            f"User's Full Review: {review}",
            "",
            "",
            "Generate a concise review title (3-10 words) capturing the main sentiment in the user's style.",
            "Respond with ONLY the review title, nothing else.",
        ])
        return "\n".join(parts)

    def _build_review_generation_prompt(self, memory, target_item, user_id):
        recent = self._select_memory(memory, target_item, "review_generation", user_id)
        parts = [
            "You are a personalized review generation system. Based on a user's review history and writing style, "
            "generate a detailed review text for a product they rated.",
            "",
            "User's Recent Review History (showing their writing style and preferences):",
        ]
        for i, inter in enumerate(recent, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        rating = target_item.get("rating", "N/A")
        parts.extend([
            "",
            "",
            "Target Product to Review:",
            self._format_interaction(target_item, self.include_metadata, include_rating=False, include_review=False),
            f"User's Rating: {rating}/5",
            "",
            "",
            "Generate a detailed review reflecting the user's writing style, typical length, and the rating sentiment.",
            "Respond with ONLY the review text, nothing else.",
        ])
        return "\n".join(parts)

    def _build_item_ranking_prompt(self, memory, candidate_asins, metadata_loader, target_item, user_id):
        if target_item is None:
            target_item = {"parent_asin": candidate_asins[0] if candidate_asins else None}
        recent = self._select_memory(memory, target_item, "item_ranking", user_id)
        parts = [
            "You are a personalized item recommendation system. Based on a user's purchase and review history, "
            "rank the candidate items in order of relevance (most likely to purchase first).",
            "",
            "User's Recent Purchase and Review History:",
        ]
        for i, inter in enumerate(recent, 1):
            parts.append(f"\n{i}. {self._format_interaction(inter, self.include_metadata)}")
        parts.extend(["", "", "Candidate Items to Rank:", ""])
        for i, asin in enumerate(candidate_asins, 1):
            meta = metadata_loader.get_metadata(asin) if metadata_loader else None
            parts.append(f"\nCandidate {i} (ASIN: {asin}):")
            if meta:
                if "title" in meta:
                    parts.append(f"  Title: {meta['title']}")
                if "average_rating" in meta:
                    parts.append(f"  Average Rating: {meta['average_rating']}")
                if "main_category" in meta:
                    parts.append(f"  Category: {meta['main_category']}")
                if meta.get("description"):
                    desc = str(meta["description"])[:200]
                    parts.append(f"  Description: {desc}...")
            else:
                parts.append("  (No metadata available)")
        parts.extend([
            "",
            "",
            "Rank ALL candidate items from most to least relevant for this user.",
            "Respond with a comma-separated list of ASINs in ranked order, nothing else.",
            "Example: B0085BEQX8, B00123ABC9, B00456DEF0",
        ])
        return "\n".join(parts)

    # ---- public predict_* ----

    def predict_rating(self, memory, target_item, user_id=None):
        if not memory:
            return 3.0, "", "No memory available"
        prompt = self._build_rating_prediction_prompt(memory, target_item, user_id)
        response = self._call_llm(prompt)
        try:
            r = float(response)
        except ValueError:
            nums = re.findall(r"\d+\.?\d*", response)
            r = float(nums[0]) if nums else 3.0
        return max(1.0, min(5.0, r)), prompt, response

    def predict_review_summary(self, memory, target_item, user_id=None):
        if not memory:
            return "", "", "No memory available"
        prompt = self._build_review_summarization_prompt(memory, target_item, user_id)
        response = self._call_llm(prompt)
        return response, prompt, response

    def predict_review_text(self, memory, target_item, user_id=None):
        if not memory:
            return "", "", "No memory available"
        prompt = self._build_review_generation_prompt(memory, target_item, user_id)
        response = self._call_llm(prompt)
        return response, prompt, response

    def predict_item_ranking(self, memory, candidate_asins, metadata_loader, target_item=None, user_id=None):
        if not memory:
            fallback = list(candidate_asins)
            random.shuffle(fallback)
            return fallback, "", "No memory available"
        prompt = self._build_item_ranking_prompt(memory, candidate_asins, metadata_loader, target_item, user_id)
        response = self._call_llm(prompt)

        ranked: List[str] = []
        seen: set = set()
        for match in re.findall(r"B[0-9A-Z]{9}", response):
            if match in candidate_asins and match not in seen:
                ranked.append(match)
                seen.add(match)
        for asin in candidate_asins:
            if asin not in seen:
                ranked.append(asin)
        if len(ranked) != len(candidate_asins):
            print(f"Warning: ranking parse mismatch ({len(ranked)} vs {len(candidate_asins)})", file=sys.stderr)
            ranked = list(candidate_asins)
        return ranked, prompt, response


# ----------------------------------------------------------------------------- #
# Memory-selection method factory
# ----------------------------------------------------------------------------- #

def build_method(
    method_name: str,
    max_memory_items: Optional[int],
    dataset_name: str,
    cache_dir: str = "cache",
) -> BaseMethod:
    """Build a memory-selection method by name. Only long_context and rag are supported."""
    if method_name == "rag":
        return RAGMethod(
            max_memory_items=max_memory_items,
            persistent_cache=True,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
        )
    if method_name == "long_context":
        return LongContextMethod(max_memory_items=max_memory_items)
    raise ValueError(f"Unknown method: {method_name!r}. Supported: long_context, rag.")


def print_cache_stats(method_name: str, method: BaseMethod) -> Optional[Dict[str, Any]]:
    """Print RAG cache stats if available; return the stats dict."""
    if method_name != "rag" or not hasattr(method, "get_cache_stats"):
        return None
    stats = method.get_cache_stats()
    print("\nCache Statistics:")
    print(f"  Hit Rate: {stats.get('hit_rate', 0):.2%}")
    print(f"  Cache Hits: {stats.get('cache_hits', 0)}")
    print(f"  Cache Misses: {stats.get('cache_misses', 0)}")
    print(f"  Disk Loads: {stats.get('disk_loads', 0)}")
    print(f"  Disk Saves: {stats.get('disk_saves', 0)}")
    if stats.get("persistent_cache_enabled"):
        print(f"  Disk Cached Users: {stats.get('disk_cached_users', 0)}")
    return stats


# ----------------------------------------------------------------------------- #
# Data loaders (read the cross-domain JSONL)
# ----------------------------------------------------------------------------- #

def _iter_user_records(input_path: Path):
    """Yield one parsed user record per line from a cross-domain JSONL file."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if line_num <= 20:
                    print(f"  Warning: JSON decode error at line {line_num}: {e}", file=sys.stderr)


def load_cross_domain_jsonl(
    input_path: Path,
    target_domain: str,
    num_test: int = 3,
    source_domains: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Load cross-domain users: memory = source domains, test = last N from target domain."""
    if verbose:
        print(f"\nLoading cross-domain data from: {input_path}", file=sys.stderr)
        print(f"Target domain: {target_domain}", file=sys.stderr)
        print(f"Num test interactions: {num_test}", file=sys.stderr)
        if source_domains is None:
            print("Source domains: all non-target domains", file=sys.stderr)
        elif not source_domains:
            print("Source domains: none (no-memory mode)", file=sys.stderr)
        else:
            print(f"Source domains: {source_domains}", file=sys.stderr)

    out: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for record in _iter_user_records(input_path):
        user_id = record.get("user_id")
        if not user_id:
            continue
        interactions_by_domain = record.get("interactions", {})
        target_inter = interactions_by_domain.get(target_domain, [])
        if not target_inter or len(target_inter) < num_test:
            continue

        target_sorted = sorted(target_inter, key=lambda x: x.get("timestamp", 0))
        test = target_sorted[-num_test:]

        if source_domains is None:
            mem_domains = [d for d in DOMAINS if d != target_domain]
        else:
            mem_domains = [d for d in source_domains if d != target_domain]

        memory: List[Dict[str, Any]] = []
        for d in mem_domains:
            for inter in interactions_by_domain.get(d, []):
                tagged = dict(inter)
                tagged["domain_category"] = d
                memory.append(tagged)
        memory.sort(key=lambda x: x.get("timestamp", 0))
        out.append((user_id, memory, test))

    if verbose:
        print(f"Loaded {len(out):,} users with sufficient data", file=sys.stderr)
    return out


def load_single_domain_from_cross_domain(
    input_path: Path,
    domain: str,
    num_test: int = 3,
    verbose: bool = True,
) -> List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Single-domain loader: reads the cross-domain JSONL but keeps only `domain` interactions."""
    if verbose:
        print(f"\nLoading single-domain data ({domain}) from: {input_path}", file=sys.stderr)

    out: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for record in _iter_user_records(input_path):
        user_id = record.get("user_id")
        if not user_id:
            continue
        interactions = record.get("interactions", {}).get(domain, [])
        if len(interactions) < num_test + 1:
            continue
        memory, test = split_last_n(interactions, num_test)
        if not memory or not test:
            continue
        out.append((user_id, memory, test))

    if verbose:
        print(f"Loaded {len(out):,} users with sufficient data", file=sys.stderr)
    return out


# ----------------------------------------------------------------------------- #
# Memory enrichment + target sanitization helpers
# ----------------------------------------------------------------------------- #

def enrich_memory_with_meta(
    memory: List[Dict[str, Any]],
    metadata_loader: MetadataLoader,
    drop_keys: Tuple[str, ...] = ("images", "videos", "negative_parent_asins"),
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for inter in memory:
        item = dict(inter)
        for k in drop_keys:
            item.pop(k, None)
        asin = inter.get("parent_asin")
        if asin:
            meta = metadata_loader.get_metadata(asin)
            if meta:
                meta_copy = dict(meta)
                meta_copy.pop("images", None)
                meta_copy.pop("videos", None)
                item["item_meta"] = meta_copy
        enriched.append(item)
    return enriched


def make_target_item(
    test_inter: Dict[str, Any],
    metadata_loader: MetadataLoader,
    drop_keys: Tuple[str, ...] = ("images", "videos"),
) -> Dict[str, Any]:
    target = dict(test_inter)
    for k in drop_keys:
        target.pop(k, None)
    asin = target.get("parent_asin")
    if asin:
        meta = metadata_loader.get_metadata(asin)
        if meta:
            meta_copy = dict(meta)
            meta_copy.pop("images", None)
            meta_copy.pop("videos", None)
            target["item_meta"] = meta_copy
    return target


# ----------------------------------------------------------------------------- #
# Result printing
# ----------------------------------------------------------------------------- #

def print_results(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS - {results['task'].upper().replace('_', ' ')}")
    print("=" * 80)
    if "domain" in results:
        print(f"Domain: {results['domain']}")
    if "target_domain" in results:
        print(f"Target Domain: {results['target_domain']}")
        print(f"Source Domains: {results.get('source_domains', 'all non-target')}")
    print(f"LLM Model: {results['llm_model']}")
    print(f"Method: {results.get('method', 'long_context')}")
    print(f"Test Split: Last {results['num_test']} interactions per user")
    print(f"Max Memory Items: {results.get('max_memory_items', 'All')}")
    print("-" * 80)
    print(f"Users Processed: {results['users_processed']:,}")
    print(f"Users Skipped: {results['users_skipped']:,}")
    print(f"Total Predictions: {results['total_predictions']:,}")
    print("-" * 80)

    task = results["task"]
    if task == "rating_prediction":
        print(f"MAE:  {results['mae']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
    elif task == "item_ranking":
        for k in results.get("k_values", [1, 3, 5]):
            if f"ndcg@{k}" in results:
                print(f"NDCG@{k}:   {results[f'ndcg@{k}']:.4f}")
                print(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
    else:
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
        print(f"BLEU-1:  {results['bleu1']:.4f}")
        print(f"BLEU-4:  {results['bleu4']:.4f}")
        if results.get("bertscore_f1", 0) > 0:
            print(f"BERTScore P:  {results['bertscore_precision']:.4f}")
            print(f"BERTScore R:  {results['bertscore_recall']:.4f}")
            print(f"BERTScore F1: {results['bertscore_f1']:.4f}")

    print("-" * 80)
    if "log_files" in results:
        print("Log Files:")
        print(f"  Predictions: {results['log_files']['predictions']}")
        print(f"  Summary:     {results['log_files']['summary']}")
    print("=" * 80)


# ----------------------------------------------------------------------------- #
# High-level task runners (used by both single- and cross-domain entry points)
# ----------------------------------------------------------------------------- #

def run_task(
    *,
    task: str,
    user_data: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]],
    metadata_loader: MetadataLoader,
    predictor: LLMPredictor,
    logger: EvaluationLogger,
    k_values: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one of the 4 tasks over a list of (user_id, memory, test) tuples."""
    if task == "rating_prediction":
        return _run_rating(user_data, metadata_loader, predictor, logger, verbose)
    if task == "review_summarization":
        return _run_review_summarization(user_data, metadata_loader, predictor, logger, verbose)
    if task == "review_generation":
        return _run_review_generation(user_data, metadata_loader, predictor, logger, verbose)
    if task == "item_ranking":
        return _run_item_ranking(user_data, metadata_loader, predictor, logger, k_values or [1, 3, 5], verbose)
    raise ValueError(f"Unknown task: {task}")


def _enrich_pair(memory, test_inter, metadata_loader):
    return enrich_memory_with_meta(memory, metadata_loader), make_target_item(test_inter, metadata_loader)


def _run_rating(user_data, metadata_loader, predictor, logger, verbose):
    evaluator = RatingPredictionEvaluator()
    users_processed = users_skipped = 0

    for user_id, memory, test in user_data:
        if not memory or not test:
            users_skipped += 1
            continue
        memory_e = enrich_memory_with_meta(memory, metadata_loader)
        for test_inter in test:
            actual = test_inter.get("rating")
            if actual is None:
                continue
            target = make_target_item(test_inter, metadata_loader)
            target.pop("rating", None)
            pred, prompt, resp = predictor.predict_rating(memory_e, target, user_id=user_id)
            evaluator.add_prediction(pred, actual)
            logger.log_prediction(
                user_id=user_id, memory=memory_e, target_item=target,
                prompt=prompt, llm_response=resp,
                prediction=pred, ground_truth=actual,
                metrics={"error": abs(pred - actual)},
            )
        users_processed += 1
        if verbose and users_processed % 10 == 0:
            print(f"  Processed {users_processed:,} users...", file=sys.stderr)

    m = evaluator.compute_metrics()
    return {"users_processed": users_processed, "users_skipped": users_skipped,
            "total_predictions": m["count"], "mae": m["mae"], "rmse": m["rmse"]}


def _run_text(user_data, metadata_loader, predictor, logger, verbose, *, summarization: bool):
    evaluator = TextGenerationEvaluator()
    users_processed = users_skipped = 0

    for user_id, memory, test in user_data:
        if not memory or not test:
            users_skipped += 1
            continue
        memory_e = enrich_memory_with_meta(memory, metadata_loader)
        for test_inter in test:
            actual_title = test_inter.get("title", "")
            actual_review = test_inter.get("text", "")
            actual_rating = test_inter.get("rating")

            if summarization:
                if not actual_title or not actual_review:
                    continue
                target = make_target_item(test_inter, metadata_loader)
                target.pop("title", None)  # don't leak ground truth
                pred, prompt, resp = predictor.predict_review_summary(memory_e, target, user_id=user_id)
                metrics = evaluator.add_prediction(pred, actual_title)
                logger.log_prediction(
                    user_id=user_id, memory=memory_e, target_item=target,
                    prompt=prompt, llm_response=resp,
                    prediction=pred, ground_truth=actual_title, metrics=metrics,
                )
            else:
                if not actual_review or actual_rating is None:
                    continue
                target = make_target_item(test_inter, metadata_loader)
                target.pop("text", None)
                target.pop("title", None)
                pred, prompt, resp = predictor.predict_review_text(memory_e, target, user_id=user_id)
                metrics = evaluator.add_prediction(pred, actual_review)
                logger.log_prediction(
                    user_id=user_id, memory=memory_e, target_item=target,
                    prompt=prompt, llm_response=resp,
                    prediction=pred, ground_truth=actual_review, metrics=metrics,
                )
        users_processed += 1
        if verbose and users_processed % 10 == 0:
            print(f"  Processed {users_processed:,} users...", file=sys.stderr)

    m = evaluator.compute_metrics()
    return {
        "users_processed": users_processed, "users_skipped": users_skipped,
        "total_predictions": m["count"],
        "rouge1": m["rouge1"], "rougeL": m["rougeL"],
        "bleu1": m["bleu1"], "bleu4": m["bleu4"],
        "bertscore_precision": m["bertscore_precision"],
        "bertscore_recall": m["bertscore_recall"],
        "bertscore_f1": m["bertscore_f1"],
    }


def _run_review_summarization(user_data, metadata_loader, predictor, logger, verbose):
    return _run_text(user_data, metadata_loader, predictor, logger, verbose, summarization=True)


def _run_review_generation(user_data, metadata_loader, predictor, logger, verbose):
    return _run_text(user_data, metadata_loader, predictor, logger, verbose, summarization=False)


def _run_item_ranking(user_data, metadata_loader, predictor, logger, k_values, verbose):
    evaluator = ItemRankingEvaluator(k_values=k_values)
    users_processed = users_skipped = skipped_no_neg = 0

    for user_id, memory, test in user_data:
        if not memory or not test:
            users_skipped += 1
            continue
        memory_e = enrich_memory_with_meta(memory, metadata_loader)
        for test_inter in test:
            gt = test_inter.get("parent_asin")
            negatives = test_inter.get("negative_parent_asins", [])
            if not gt or not negatives:
                skipped_no_neg += 1
                continue

            candidates = [gt] + list(negatives)
            random.shuffle(candidates)

            target = make_target_item(test_inter, metadata_loader)
            target.pop("negative_parent_asins", None)

            ranked, prompt, resp = predictor.predict_item_ranking(
                memory_e, candidates, metadata_loader, target_item=target, user_id=user_id
            )
            per_pred = {f"ndcg@{k}": evaluator._ndcg_at_k(ranked, gt, k) for k in k_values}
            per_pred.update({f"recall@{k}": evaluator._recall_at_k(ranked, gt, k) for k in k_values})

            logger.log_prediction(
                user_id=user_id, memory=memory_e,
                target_item={"candidate_asins": candidates, "parent_asin": gt},
                prompt=prompt, llm_response=resp,
                prediction=ranked, ground_truth=gt, metrics=per_pred,
            )
            evaluator.add_prediction(ranked, gt)
        users_processed += 1
        if verbose and users_processed % 10 == 0:
            print(f"  Processed {users_processed:,} users...", file=sys.stderr)

    if verbose and skipped_no_neg > 0:
        print(f"Warning: skipped {skipped_no_neg} test interactions without negative samples", file=sys.stderr)

    m = evaluator.compute_metrics()
    result = {
        "users_processed": users_processed, "users_skipped": users_skipped,
        "skipped_no_negatives": skipped_no_neg,
        "total_predictions": m["count"], "k_values": k_values,
    }
    for k in k_values:
        result[f"ndcg@{k}"] = m[f"ndcg@{k}"]
        result[f"recall@{k}"] = m[f"recall@{k}"]
    return result
