#!/usr/bin/env python3
"""
Cross-domain evaluation entry point for MemoryCD.

Reads the unified cross-domain JSONL where each user has interactions grouped
by domain. For one --target-domain:
- Test set: last `--num-test` interactions of the target domain (by timestamp).
- Memory:   interactions from --source-domains (default: all non-target domains),
            sorted globally by timestamp, tagged with `domain_category`.
- Use --no-memory (or --source-domains with no values) to run the no-memory baseline.

Tasks (set with --task): rating_prediction, review_summarization,
review_generation, item_ranking. Methods (--method): long_context, rag.

Usage examples:
    # Cross-domain default (memory = the 3 non-target domains)
    python evaluation_all_4task_cross_domain.py \\
        --task rating_prediction --target-domain Home_and_Kitchen \\
        --input data/cross_domain.jsonl --llm-model openai/gpt-5

    # Limit memory to a subset of source domains
    python evaluation_all_4task_cross_domain.py \\
        --task review_summarization --target-domain Home_and_Kitchen \\
        --source-domains Books Electronics --llm-model openai/gpt-5

    # No-memory baseline
    python evaluation_all_4task_cross_domain.py \\
        --task rating_prediction --target-domain Books \\
        --no-memory --llm-model openai/gpt-5
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
    load_cross_domain_jsonl,
    print_cache_stats,
    print_results,
    run_task,
)


_DOMAIN_ABBREV = {
    "Beauty_and_Personal_Care": "B",
    "Books": "Bo",
    "Electronics": "E",
    "Home_and_Kitchen": "H",
}


def _source_abbrev(source_domains: Optional[List[str]]) -> str:
    if source_domains is None:
        return "srcAll"
    if not source_domains:
        return "noMem"
    return "src" + "".join(_DOMAIN_ABBREV.get(d, d[:2]) for d in sorted(source_domains))


def evaluate(
    *,
    task: str,
    target_domain: str,
    source_domains: Optional[List[str]],
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
    metadata_loader = MetadataLoader(meta_dir, domains=DOMAINS)
    user_data = load_cross_domain_jsonl(
        input_path,
        target_domain=target_domain,
        num_test=num_test,
        source_domains=source_domains,
        verbose=verbose,
    )
    if not user_data:
        print("Error: no users with sufficient data loaded", file=sys.stderr)
        return {}
    if max_users is not None and max_users > 0:
        user_data = user_data[:max_users]

    method = build_method(method_name, max_memory_items, dataset_name=f"cross_{target_domain}")
    predictor = LLMPredictor(model=llm_model, method=method)

    domain_part = f"{target_domain}_{_source_abbrev(source_domains)}"
    run_id = build_run_id(
        setting="cross", task_name=task, method_name=method_name,
        model_name=llm_model, domain_part=domain_part,
        max_users=max_users, max_memory_items=max_memory_items, num_test=num_test,
    )
    logger = EvaluationLogger(log_dir / target_domain, run_id=run_id)

    if verbose:
        srcs = source_domains if source_domains is not None else "all non-target"
        print(f"\n{'=' * 80}\nTASK: {task} (cross-domain target={target_domain}, sources={srcs})\n{'=' * 80}")

    task_result = run_task(
        task=task, user_data=user_data, metadata_loader=metadata_loader,
        predictor=predictor, logger=logger, k_values=k_values, verbose=verbose,
    )

    results: Dict[str, Any] = {
        "task": task,
        "setting": "cross_domain",
        "target_domain": target_domain,
        "source_domains": source_domains if source_domains is not None else "all_non_target",
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
        description="Cross-domain LLM evaluation for the 4 MemoryCD tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True,
                        choices=["rating_prediction", "review_summarization", "review_generation", "item_ranking"])
    parser.add_argument("--target-domain", required=True, choices=DOMAINS,
                        help="Domain whose last N interactions form the test set.")
    parser.add_argument("--source-domains", type=str, nargs="*", default=None, choices=DOMAINS,
                        help="Domains used for memory (default: all non-target domains).")
    parser.add_argument("--no-memory", action="store_true",
                        help="Use no memory (overrides --source-domains).")
    parser.add_argument("--input", type=Path, default=Path("data/cross_domain.jsonl"))
    parser.add_argument("--meta-dir", type=Path, default=Path("meta"))
    parser.add_argument("--llm-model", type=str, default="openai/gpt-5")
    parser.add_argument("--method", type=str, default="long_context",
                        choices=["long_context", "rag"])
    parser.add_argument("--num-test", type=int, default=3)
    parser.add_argument("--max-memory-items", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=None,
                        help="K values for NDCG@K / Recall@K (default: 1 3 5). Only for item_ranking.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/cross"))
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    if not args.input.exists():
        print(f"Error: input JSONL not found: {args.input}", file=sys.stderr)
        return 1

    if args.no_memory:
        source_domains: Optional[List[str]] = []
    else:
        source_domains = args.source_domains  # may be None (=> all non-target)

    results = evaluate(
        task=args.task,
        target_domain=args.target_domain,
        source_domains=source_domains,
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
