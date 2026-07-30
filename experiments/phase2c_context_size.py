"""Phase 2c — context size, and position within the context.

Two separate questions that the original single sweep conflated:

  (a) Does adding more retrieved chunks help or hurt? Sweeping K=1..10 changes two
      things at once — more evidence *and* more distraction — so K alone cannot
      attribute a drop to either.

  (b) Does the *position* of the gold chunk inside a fixed-size context matter?
      This is the actual "lost in the middle" claim. It is tested at fixed K=8 by
      moving the same gold chunk to the first, middle, and last slot, so the
      context contents are identical across conditions and only order changes.

The position sweep is restricted to items whose gold document was retrieved within
the top-8; on the others there is no gold chunk to place and the condition is
undefined.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import noise as noise_mod
from utils.metrics import ABSTAIN, CORRECT, UNSUPPORTED
from utils.pipeline import (announce_budget, quota_guard, build_retrieval_cache, common_args,
                            load_dataset, run_condition, run_manifest, save_results)
from utils.stats import paired_diff_by_qid, rate_with_ci_clustered

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10]
POSITION_K = 8
POSITIONS = [0, POSITION_K // 2, POSITION_K - 1]


def main():
    args = common_args("Phase 2c: context size and gold position sweeps")
    qa_pairs, corpus, store, cache_path = load_dataset(
        args.dataset, limit=args.limit, rebuild_index=args.rebuild_index
    )
    retrieval = build_retrieval_cache(qa_pairs, store, cache_path)

    announce_budget(len(K_VALUES) + len(POSITIONS), len(qa_pairs),
                    use_judge=not args.no_judge, repeats=args.repeats)

    run_kw = dict(use_judge=not args.no_judge, model_name=args.model,
                  repeats=args.repeats, temperature=args.temperature,
                  dataset=args.dataset)

    # ---- (a) context size ------------------------------------------------- #
    print(f"--- Phase 2c(a): K sweep on {args.dataset} ---")
    size_conditions, size_rows = {}, {}
    for k in K_VALUES:
        name = f"k={k}"
        rows, summary = run_condition(
            name, qa_pairs, retrieval, corpus, k=k,
            noise_type=noise_mod.CLEAN, **run_kw,
        )
        valid = [r for r in rows if r["outcome"] != "INVALID"]
        answerable = [r for r in valid if r["is_answerable"]]
        summary["ci"] = {
            "F1": rate_with_ci_clustered(answerable, lambda r: r["f1"]),
            "abstention_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == ABSTAIN),
            "hallucination_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == UNSUPPORTED),
        }
        size_conditions[name], size_rows[name] = summary, rows
        print(f"  K={k:>2}: F1={summary.get('F1')} P@K={summary.get('gold_in_context_rate')} "
              f"abstain={summary.get('abstention_rate')} halluc={summary.get('hallucination_rate')}")

    # ---- (b) gold position at fixed K ------------------------------------- #
    eligible = [qa for qa in qa_pairs
                if _gold_within(retrieval.get(qa["qid"], []), qa, POSITION_K)]
    print(f"\n--- Phase 2c(b): gold position at K={POSITION_K} "
          f"({len(eligible)}/{len(qa_pairs)} items have gold in top-{POSITION_K}) ---")

    pos_conditions, pos_rows = {}, {}
    for pos in POSITIONS:
        name = f"pos={pos}"
        rows, summary = run_condition(
            name, eligible, retrieval, corpus, k=POSITION_K,
            noise_type=noise_mod.CLEAN, gold_position=pos, **run_kw,
        )
        answerable = [r for r in rows if r["outcome"] != "INVALID" and r["is_answerable"]]
        summary["ci"] = {
            "F1": rate_with_ci_clustered(answerable, lambda r: r["f1"]),
            "correct_rate": rate_with_ci_clustered(answerable, lambda r: r["outcome"] == CORRECT),
        }
        pos_conditions[name], pos_rows[name] = summary, rows
        print(f"  gold at slot {pos}: F1={summary.get('F1')} "
              f"correct={summary['ci']['correct_rate']['mean']}")

    first = pos_rows.get(f"pos={POSITIONS[0]}")
    middle = pos_rows.get(f"pos={POSITIONS[1]}")
    if first and middle:
        paired = _paired_f1(first, middle)
        print(f"\n  middle vs first, paired dF1 = {paired['diff']} "
              f"[{paired['lo']}, {paired['hi']}] p={paired['p_two_sided']}")
        if paired["hi"] is not None and paired["hi"] < 0:
            print("  -> significant drop when gold sits mid-context (lost-in-the-middle)")
        else:
            print("  -> no significant position effect at this n")

    save_results(f"phase2c_context_{args.dataset}.json", {
        "manifest": run_manifest(args.dataset, qa_pairs, {
            "k_values": K_VALUES, "position_k": POSITION_K, "positions": POSITIONS,
            "n_position_eligible": len(eligible),
        }),
        "size_conditions": size_conditions,
        "position_conditions": pos_conditions,
        "rows": {"size": size_rows, "position": pos_rows},
    })


def _gold_within(ranked, qa, k):
    gold = {g for g in (qa.get("gold_doc_ids") or [qa.get("gold_doc_id")]) if g}
    return any(c["doc_id"] in gold for c in ranked[:k])


def _paired_f1(rows_a, rows_b):
    return paired_diff_by_qid(rows_a, rows_b, lambda r: r["f1"])


if __name__ == "__main__":
    quota_guard(main)()
