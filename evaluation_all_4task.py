#!/usr/bin/env python3
"""
Single-domain evaluation entry point for MemoryCD.

Reads the unified cross-domain JSONL but evaluates within one domain at a time:
the memory and test interactions both come from `--domain`. The four tasks are:

1. Personalized rating prediction       (MAE, RMSE)
2. Personalized review summarization    (ROUGE-1/L, BLEU-1/4, BERTScore)
3. Personalized review generation       (ROUGE-1/L, BLEU-1/4, BERTScore)
4. Personalized item ranking            (NDCG@K, Recall@K)

For each user the interactions in `--domain` are sorted by timestamp:
- last `--num-test` interactions => test set
- earlier interactions           => memory

Supported memory-selection methods (set via --method):
- long_context : keep most-recent N memory items
- rag          : BM25 retrieval over memory items (uses `--max-memory-items`)

Usage examples:
    python evaluation_all_4task.py \\
        --task rating_prediction --domain Books \\
        --input data/cross_domain.jsonl \\
        --llm-model openai/gpt-5

    python evaluation_all_4task.py \\
        --task item_ranking --domain Electronics \\
        --input data/cross_domain.jsonl \\
        --method rag --max-memory-items 10 --k-values 1 3 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval_core import (
    DOMAINS,
    EvaluationLogger,
    LLMPredictor,
    MetadataLoader,
    build_method,
    build_run_id,
    load_single_domain_from_cross_domain,
    print_cache_stats,
    print_results,
    run_task,
)


def evaluate(
    *,
    task: str,
    domain: str,
    input_path: Path,
    meta_dir: Path,
    llm_model: str,
    method_name: str,
    max_memory_items: Optional[int],
    max_users: Optional[int],
    num_test: int,
    k_values: Optional[List[int]],
    log_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    metadata_loader = MetadataLoader(meta_dir, domains=[domain])
    user_data = load_single_domain_from_cross_domain(input_path, domain, num_test, verbose=verbose)
    if not user_data:
        print("Error: no users with sufficient data loaded", file=sys.stderr)
        return {}
    if max_users is not None and max_users > 0:
        user_data = user_data[:max_users]

    method = build_method(method_name, max_memory_items, dataset_name=domain)
    predictor = LLMPredictor(model=llm_model, method=method)

    run_id = build_run_id(
        setting="single", task_name=task, method_name=method_name,
        model_name=llm_model, domain_part=domain,
        max_users=max_users, max_memory_items=max_memory_items, num_test=num_test,
    )
    logger = EvaluationLogger(log_dir / domain, run_id=run_id)

    if verbose:
        print(f"\n{'=' * 80}\nTASK: {task} (single-domain={domain})\n{'=' * 80}")

    task_result = run_task(
        task=task, user_data=user_data, metadata_loader=metadata_loader,
        predictor=predictor, logger=logger, k_values=k_values, verbose=verbose,
    )

    results: Dict[str, Any] = {
        "task": task,
        "setting": "single_domain",
        "domain": domain,
        "llm_model": llm_model,
        "method": method_name,
        "num_test": num_test,
        "max_memory_items": max_memory_items,
        **task_result,
    }
    stats = print_cache_stats(method_name, method)
    if stats is not None:
        results["cache_stats"] = stats

    results["log_files"] = {
        "predictions": str(logger.predictions_log_path),
        "summary": str(logger.summary_log_path),
    }
    logger.log_summary(results)
    logger.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-domain LLM evaluation for the 4 MemoryCD tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True,
                        choices=["rating_prediction", "review_summarization", "review_generation", "item_ranking"])
    parser.add_argument("--domain", required=True, choices=DOMAINS,
                        help="Domain whose interactions are used for both memory and test.")
    parser.add_argument("--input", type=Path, default=Path("data/cross_domain.jsonl"),
                        help="Path to the unified cross-domain JSONL (default: data/cross_domain.jsonl).")
    parser.add_argument("--meta-dir", type=Path, default=Path("meta"),
                        help="Directory containing meta_<Domain>.jsonl[.gz] files (default: meta).")
    parser.add_argument("--llm-model", type=str, default="openai/gpt-5")
    parser.add_argument("--method", type=str, default="long_context",
                        choices=["long_context", "rag"])
    parser.add_argument("--num-test", type=int, default=3,
                        help="Number of last interactions per user used as test (default: 3).")
    parser.add_argument("--max-memory-items", type=int, default=None,
                        help="Cap on memory items (default: unlimited for long_context; top-K for rag).")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=None,
                        help="K values for NDCG@K / Recall@K (default: 1 3 5). Only for item_ranking.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/single"))
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    if not args.input.exists():
        print(f"Error: input JSONL not found: {args.input}", file=sys.stderr)
        return 1

    results = evaluate(
        task=args.task,
        domain=args.domain,
        input_path=args.input,
        meta_dir=args.meta_dir,
        llm_model=args.llm_model,
        method_name=args.method,
        max_memory_items=args.max_memory_items,
        max_users=args.max_users,
        num_test=args.num_test,
        k_values=args.k_values,
        log_dir=args.log_dir,
        verbose=verbose,
    )
    if not results:
        return 1
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
