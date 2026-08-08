"""Phase 2b — the core experiment: how generation degrades as context degrades.

Grid: noise type x noise fraction, at a fixed context size K=4 so that a fraction
maps onto whole slots (0 / 1 / 2 / 3 / 4 of 4).

H1 (the non-obvious one): the relationship between noise fraction and *silent*
failure is non-monotonic. At 100% noise the model has nothing to work with and
abstains; at 50% it has enough plausible material to commit to a confident wrong
answer. If true, partially-broken retrieval is more dangerous than fully-broken
retrieval, which inverts the usual "improve retrieval quality" intuition.

H2: RELATED noise (adjacent, plausible, wrong) suppresses abstention more than
RANDOM noise at the same fraction — the model uses topical proximity as evidence.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import noise as noise_mod
from utils.metrics import ABSTAIN, CORRECT, UNSUPPORTED
from utils.pipeline import (announce_budget, quota_guard, build_retrieval_cache, common_args,
                            load_dataset, run_condition, run_manifest, save_results)
from utils.stats import apply_fdr, paired_diff_by_qid, rate_with_ci_clustered

K = 4

# RELATED is swept finely because H1 lives on that curve; the other noise types get
# the endpoints needed to test H2 without quadrupling the API bill.
GRID = (
    [(noise_mod.RELATED, f) for f in (0.25, 0.5, 0.75, 1.0)]
    + [(noise_mod.RANDOM, f) for f in (0.5, 1.0)]
    + [(noise_mod.COUNTERFACTUAL, f) for f in (0.5, 1.0)]
    + [(noise_mod.DROP_GOLD, 1.0)]
)


def main():
    args = common_args("Phase 2b: noise type x fraction sweep")
    qa_pairs, corpus, store, cache_path = load_dataset(
        args.dataset, limit=args.limit, rebuild_index=args.rebuild_index
    )
    retrieval = build_retrieval_cache(qa_pairs, store, cache_path)

    announce_budget(len(GRID) + 1, len(qa_pairs), use_judge=not args.no_judge,
                    repeats=args.repeats)
    print(f"--- Phase 2b: Noise sweep (K={K}) on {args.dataset} ---")

    run_kw = dict(use_judge=not args.no_judge, model_name=args.model,
                  repeats=args.repeats, temperature=args.temperature,
                  dataset=args.dataset)
    conditions, all_rows = {}, {}

    control_rows, control_summary = run_condition(
        "clean", qa_pairs, retrieval, corpus, k=K,
        noise_type=noise_mod.CLEAN, **run_kw,
    )
    conditions["clean"], all_rows["clean"] = control_summary, control_rows
    print(f"  clean: {_line(control_summary)}")

    for noise_type, frac in GRID:
        name = f"{noise_type}@{int(frac * 100)}"
        rows, summary = run_condition(
            name, qa_pairs, retrieval, corpus, k=K,
            noise_type=noise_type, noise_frac=frac, **run_kw,
        )

        valid = [r for r in rows if r["outcome"] != "INVALID"]
        summary["ci"] = {
            "abstention_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == ABSTAIN),
            "hallucination_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == UNSUPPORTED),
            "correct_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == CORRECT),
            "silent_failure_rate": rate_with_ci_clustered(
                [r for r in valid if r["outcome"] != CORRECT],
                lambda r: r["outcome"] != ABSTAIN),
        }
        # Paired against the clean control on the same questions.
        summary["vs_clean"] = _paired(control_rows, rows)

        conditions[name], all_rows[name] = summary, rows
        print(f"  {name}: {_line(summary)}")

        # Write after every condition, not only at the end. On free-tier quota this
        # sweep spans days; a results file that only appears once all ten cells are
        # done means a partially-collected study produces nothing downstream --
        # no tables, no figures, no Phase 3 -- despite having the rows on disk.
        _write(args, conditions, all_rows, partial=True)

    # Every noise condition is tested against the same clean control, so this is a
    # family of ~9 comparisons per metric. Uncorrected, one "significant" cell is
    # expected by chance alone. FDR-adjust before anything is called a finding.
    for metric in ("F1", "abstention_rate", "hallucination_rate"):
        family = {name: (conditions[name].get("vs_clean") or {}).get(metric)
                  for name in conditions if name != "clean"}
        apply_fdr(family)
    n_survive = sum(
        1 for name in conditions if name != "clean"
        and ((conditions[name].get("vs_clean") or {}).get("hallucination_rate") or {})
        .get("significant_fdr")
    )
    print(f"\nFDR (BH, alpha=0.05) over {len(conditions) - 1} comparisons: "
          f"{n_survive} hallucination-rate differences survive correction.")

    _write(args, conditions, all_rows, partial=False)
    _report_h1(conditions)
    _report_h2(conditions)


def _write(args, conditions, all_rows, partial):
    """Persist the sweep. `partial` marks a run that has not covered the full grid."""
    done = [n for n in conditions if n != "clean"]
    save_results(f"phase2b_noise_{args.dataset}.json", {
        "manifest": run_manifest(args.dataset, args_qa(all_rows), {
            "k": K,
            "grid": [list(g) for g in GRID],
            "grid_complete": not partial and len(done) == len(GRID),
            "conditions_collected": len(conditions),
            "conditions_planned": len(GRID) + 1,
        }),
        "hypotheses": {
            "H1": "silent-failure rate is non-monotonic in noise fraction "
                  "(peaks at partial noise, falls at total noise)",
            "H2": "related noise suppresses abstention more than random noise "
                  "at equal fraction",
        },
        "conditions": conditions,
        "rows": all_rows,
    })


def args_qa(all_rows):
    """Reconstruct the question set from collected rows for the manifest."""
    seen, qa = set(), []
    for rows in all_rows.values():
        for r in rows:
            if r.get("qid") not in seen:
                seen.add(r.get("qid"))
                qa.append({"qid": r.get("qid"), "is_answerable": r.get("is_answerable", True)})
    return qa


def _line(s):
    return (f"F1={s.get('F1')} abstain={s.get('abstention_rate')} "
            f"halluc={s.get('hallucination_rate')} silent={s.get('silent_failure_rate')}")


def _paired(control_rows, rows):
    return {
        "F1": paired_diff_by_qid(control_rows, rows, lambda r: r["f1"]),
        "abstention_rate": paired_diff_by_qid(
            control_rows, rows, lambda r: r["outcome"] == ABSTAIN),
        "hallucination_rate": paired_diff_by_qid(
            control_rows, rows, lambda r: r["outcome"] == UNSUPPORTED),
    }


def _report_h1(conditions):
    curve = [(f, conditions.get(f"related@{int(f * 100)}", {}).get("silent_failure_rate"))
             for f in (0.25, 0.5, 0.75, 1.0)]
    curve = [(f, v) for f, v in curve if v is not None]
    print("\nH1 — silent-failure rate vs RELATED noise fraction:")
    for f, v in curve:
        print(f"  {int(f * 100):>3}%  {v}")
    if len(curve) >= 3:
        peak_f, peak_v = max(curve, key=lambda x: x[1])
        interior = peak_f not in (curve[0][0], curve[-1][0])
        print(f"  peak at {int(peak_f * 100)}% ({peak_v}) — "
              f"{'interior peak, consistent with H1' if interior else 'monotonic, H1 not supported'}")


def _report_h2(conditions):
    print("\nH2 — abstention rate, related vs random at equal fraction:")
    for f in (0.5, 1.0):
        rel = conditions.get(f"related@{int(f * 100)}", {}).get("abstention_rate")
        rnd = conditions.get(f"random@{int(f * 100)}", {}).get("abstention_rate")
        if rel is not None and rnd is not None:
            print(f"  {int(f * 100):>3}%  related={rel}  random={rnd}  "
                  f"{'H2 direction holds' if rel < rnd else 'H2 direction does not hold'}")


if __name__ == "__main__":
    quota_guard(main)()
