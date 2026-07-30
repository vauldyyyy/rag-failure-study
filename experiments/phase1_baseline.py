"""Phase 1 — control condition.

Clean retrieval, K=3, default prompt. Every later phase is measured as a paired
difference against this run, so it is the only place the absolute numbers matter.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import noise as noise_mod
from utils.metrics import ABSTAIN, CORRECT, UNSUPPORTED
from utils.pipeline import (announce_budget, build_retrieval_cache, common_args,
                            load_dataset, run_condition, run_manifest, save_results)
from utils.stats import rate_with_ci_clustered


def main():
    args = common_args("Phase 1: clean-retrieval control condition")
    qa_pairs, corpus, store, cache_path = load_dataset(
        args.dataset, limit=args.limit, rebuild_index=args.rebuild_index
    )
    retrieval = build_retrieval_cache(qa_pairs, store, cache_path)

    announce_budget(1, len(qa_pairs), use_judge=not args.no_judge, repeats=args.repeats)
    print(f"--- Phase 1: Baseline (K={config.TOP_K}, clean) on {args.dataset} ---")

    rows, summary = run_condition(
        "baseline", qa_pairs, retrieval, corpus,
        k=config.TOP_K, noise_type=noise_mod.CLEAN,
        use_judge=not args.no_judge, model_name=args.model,
        repeats=args.repeats, temperature=args.temperature,
    )

    valid = [r for r in rows if r["outcome"] != "INVALID"]
    answerable = [r for r in valid if r["is_answerable"]]
    unanswerable = [r for r in valid if not r["is_answerable"]]

    summary["ci"] = {
        "F1": rate_with_ci_clustered(answerable, lambda r: r["f1"]),
        "EM": rate_with_ci_clustered(answerable, lambda r: r["em"]),
        "P@K": rate_with_ci_clustered(valid, lambda r: r["p_at_k"]),
        "abstention_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == ABSTAIN),
        "hallucination_rate_unanswerable": rate_with_ci_clustered(
            unanswerable, lambda r: r["outcome"] == UNSUPPORTED),
        "correct_rate_answerable": rate_with_ci_clustered(
            answerable, lambda r: r["outcome"] == CORRECT),
    }

    save_results(f"phase1_baseline_{args.dataset}.json", {
        "manifest": run_manifest(args.dataset, qa_pairs),
        "summary": summary,
        "rows": rows,
    })

    print("\nBaseline:")
    for key in ("EM", "F1", "abstention_rate", "hallucination_rate_unanswerable",
                "gold_in_context_rate", "parametric_correct_rate", "silent_failure_rate"):
        print(f"  {key:34s} {summary.get(key)}")


if __name__ == "__main__":
    main()
