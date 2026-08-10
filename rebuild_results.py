"""Reconstruct a phase's results file from its checkpoints, without re-running.

Needed because a quota-limited sweep can accumulate hundreds of scored rows across
several days while never reaching the end of its grid -- and until this existed,
that meant no results file, and therefore no tables, no figures and no Phase 3,
despite the data sitting on disk.

Costs nothing: it only re-aggregates rows that were already generated and scored.

    python rebuild_results.py --phase 2b
    python rebuild_results.py --phase 2b --dataset squad
"""

import argparse
import glob
import json
import os
import sys

import config
from utils.metrics import ABSTAIN, CORRECT, UNSUPPORTED, aggregate
from utils.pipeline import save_results
from utils.stats import apply_fdr, paired_diff_by_qid, rate_with_ci_clustered

# Which checkpoint names belong to which phase, and the order to present them in.
PHASE_CONDITIONS = {
    "2b": ["clean", "related@25", "related@50", "related@75", "related@100",
           "random@50", "random@100", "counterfactual@50", "counterfactual@100",
           "drop_gold@100"],
    "1": ["baseline"],
}


def ckpt_name(condition):
    return condition.replace("/", "_").replace("@", "_at_").replace("=", "")


def load_condition(dataset, condition):
    path = os.path.join(config.RESULTS_DIR, "checkpoints",
                        f"{dataset}_{ckpt_name(condition)}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows or None


def rebuild_2b(dataset, expected_n):
    conditions, all_rows, incomplete = {}, {}, []

    for name in PHASE_CONDITIONS["2b"]:
        rows = load_condition(dataset, name)
        if not rows:
            continue
        if expected_n and len(rows) < expected_n:
            incomplete.append(f"{name} ({len(rows)}/{expected_n})")

        valid = [r for r in rows if r["outcome"] != "INVALID"]
        summary = aggregate(rows)
        noise_type, _, frac = name.partition("@")
        summary.update({
            "condition": name,
            "noise_type": "clean" if name == "clean" else noise_type,
            "noise_frac": (int(frac) / 100.0) if frac.isdigit() else 0.0,
            "k": 4,
            "n_rows_collected": len(rows),
            "complete": not (expected_n and len(rows) < expected_n),
            "ci": {
                "abstention_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == ABSTAIN),
                "hallucination_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == UNSUPPORTED),
                "correct_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == CORRECT),
                "silent_failure_rate": rate_with_ci_clustered(
                    [r for r in valid if r["outcome"] != CORRECT],
                    lambda r: r["outcome"] != ABSTAIN),
            },
        })
        conditions[name], all_rows[name] = summary, rows

    if "clean" not in conditions:
        sys.exit("No 'clean' control checkpoint; cannot compute paired differences.")

    control = all_rows["clean"]
    for name in conditions:
        if name == "clean":
            continue
        conditions[name]["vs_clean"] = {
            "F1": paired_diff_by_qid(control, all_rows[name], lambda r: r["f1"]),
            "correct_rate": paired_diff_by_qid(control, all_rows[name],
                                               lambda r: r["outcome"] == CORRECT),
            "abstention_rate": paired_diff_by_qid(control, all_rows[name],
                                                  lambda r: r["outcome"] == ABSTAIN),
            "hallucination_rate": paired_diff_by_qid(control, all_rows[name],
                                                     lambda r: r["outcome"] == UNSUPPORTED),
            "incorrect_grounded_rate": paired_diff_by_qid(
                control, all_rows[name], lambda r: r["outcome"] == "INCORRECT_GROUNDED"),
        }

    # FDR across the family of conditions-vs-control, per metric.
    for metric in ("F1", "correct_rate", "abstention_rate", "hallucination_rate",
                   "incorrect_grounded_rate"):
        apply_fdr({n: conditions[n]["vs_clean"].get(metric)
                   for n in conditions if n != "clean"})

    return conditions, all_rows, incomplete


def main():
    p = argparse.ArgumentParser(description="Rebuild results files from checkpoints")
    p.add_argument("--phase", default="2b", choices=["2b"])
    p.add_argument("--dataset", default="squad", choices=["squad", "hotpot"])
    p.add_argument("--expected-n", type=int, default=150,
                   help="Rows per condition in a complete run; used to flag partials")
    args = p.parse_args()

    conditions, all_rows, incomplete = rebuild_2b(args.dataset, args.expected_n)
    planned = len(PHASE_CONDITIONS["2b"])

    manifest = {
        "dataset": args.dataset,
        "rebuilt_from_checkpoints": True,
        "conditions_collected": len(conditions),
        "conditions_planned": planned,
        "grid_complete": len(conditions) == planned and not incomplete,
        "incomplete_conditions": incomplete,
        "llm_model": config.LLM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "judge_self_eval": config.JUDGE_SELF_EVAL,
        "embed_backend": config.EMBED_BACKEND,
        "embed_model": config.LOCAL_EMBED_MODEL if config.EMBED_BACKEND == "local" else config.EMBED_MODEL,
        "seed": config.SEED,
        "k": 4,
        "n_qa": max((c["n_rows_collected"] for c in conditions.values()), default=0),
        "n_answerable": sum(1 for r in all_rows.get("clean", []) if r.get("is_answerable")),
        "n_unanswerable": sum(1 for r in all_rows.get("clean", []) if not r.get("is_answerable")),
    }

    save_results(f"phase2b_noise_{args.dataset}.json", {
        "manifest": manifest,
        "hypotheses": {
            "H1": "silent-failure rate is non-monotonic in noise fraction",
            "H2": "related noise suppresses abstention more than random at equal fraction",
        },
        "conditions": conditions,
        "rows": all_rows,
    })

    print(f"\nRebuilt {len(conditions)}/{planned} conditions "
          f"({sum(len(r) for r in all_rows.values())} scored rows).")
    if incomplete:
        print(f"Partial: {', '.join(incomplete)}")
    if len(conditions) < planned:
        missing = [c for c in PHASE_CONDITIONS["2b"] if c not in conditions]
        print(f"Not yet collected: {', '.join(missing)}")
    print("\nThe results file is marked grid_complete="
          f"{manifest['grid_complete']}; anything built from it inherits that flag.")


if __name__ == "__main__":
    main()
