"""Phase 3 — can failure be predicted before generation?

The second half of the research question. Phases 1-2 measure *when* RAG degrades;
this asks whether a system could have known in advance, using only signals that
exist before the generator is called (similarity scores and their shape).

If a retrieval-side signal separates failures from successes well enough, a
production system can gate on it and abstain instead of guessing — which is
exactly the mitigation Phase 4 cannot deliver through prompting alone.

Costs nothing: it re-analyses rows already written by phases 1, 2b and 2c.
"""

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.metrics import ABSTAIN, CORRECT, UNSUPPORTED
from utils.pipeline import save_results
from utils.stats import auroc

SIGNALS = ["top1_score", "mean_score", "min_score", "score_margin", "score_spread"]


def collect_rows(dataset):
    """Flatten every row this study has produced for `dataset`."""
    rows = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, f"phase*_{dataset}.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        source = os.path.basename(path)
        for row in _walk(payload.get("rows")):
            row["_source"] = source
            rows.append(row)
    return [r for r in rows if r.get("outcome") != "INVALID" and r.get("signals")]


def _walk(node):
    if node is None:
        return
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "outcome" in item:
                yield item
            else:
                yield from _walk(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)


def main():
    p = argparse.ArgumentParser(description="Phase 3: pre-generation failure prediction")
    p.add_argument("--dataset", default="squad", choices=["squad", "hotpot"])
    p.add_argument("--dry-run", action="store_true", help="Analyse results_dryrun/ instead")
    args = p.parse_args()
    config.set_dry_run(args.dry_run)

    rows = collect_rows(args.dataset)
    if not rows:
        print(f"No usable rows for '{args.dataset}'. Run phases 1/2b/2c first.")
        return

    print(f"--- Phase 3: {len(rows)} scored generations from "
          f"{len({r['_source'] for r in rows})} result files ---\n")

    # Two prediction targets, because they call for opposite system responses:
    #   any_failure  -> the answer was not correct (abstention included)
    #   silent_wrong -> the answer was wrong AND asserted (abstention excluded).
    #                   This is the one worth gating on: it is the harmful case.
    targets = {
        "any_failure": lambda r: 0 if r["outcome"] == CORRECT else 1,
        "silent_wrong": lambda r: 1 if r["outcome"] == UNSUPPORTED else 0,
    }

    report = {"n_rows": len(rows), "targets": {}}
    for target_name, label_fn in targets.items():
        labels = [label_fn(r) for r in rows]
        base = sum(labels) / len(labels)
        print(f"{target_name}  (base rate {base:.3f})")

        scores = {}
        for signal in SIGNALS:
            values = [r["signals"].get(signal, 0.0) for r in rows]
            # Higher similarity should mean lower failure risk, so negate to keep
            # AUROC oriented as "higher score = more likely to fail".
            a = auroc([-v for v in values], labels)
            scores[signal] = a
            print(f"  {signal:16s} AUROC {a}")

        best = max((s for s in scores.items() if s[1] is not None),
                   key=lambda kv: kv[1], default=(None, None))
        report["targets"][target_name] = {
            "base_rate": round(base, 4),
            "auroc": scores,
            "best_signal": best[0],
            "best_auroc": best[1],
            "threshold_table": _thresholds(rows, best[0], labels) if best[0] else [],
        }
        if best[0]:
            print(f"  best: {best[0]} (AUROC {best[1]})")
            print(f"  {'gate':<9}{'keep%':<9}{'fail|kept':<12}{'fail|dropped'}")
            for t in report["targets"][target_name]["threshold_table"]:
                print(f"  {t['threshold']:<9.4f}{t['kept_frac']:<9.3f}"
                      f"{_fmt(t['failure_rate_kept']):<12}{_fmt(t['failure_rate_dropped'])}")
        print()

    # AUROC alone does not tell a practitioner whether to deploy a gate. This does.
    report["gate_economics"] = gate_economics(rows)

    interp = (
        "AUROC near 0.5 means retrieval scores carry no usable pre-generation signal; "
        "a system cannot gate on them and must fall back to post-hoc verification. "
        "AUROC above ~0.7 means an abstention gate is viable and the threshold table "
        "gives its operating points. The break-even cost ratio below translates that "
        "into a deployment decision."
    )
    report["interpretation"] = interp
    print(interp)

    save_results(f"phase3_predictability_{args.dataset}.json", report)


def gate_economics(rows, signal=None, cost_ratios=(1, 2, 5, 10, 20, 50)):
    """When is a similarity gate actually worth deploying?

    AUROC is a ranking statistic; it does not say whether to ship the gate. The
    decision depends on how much worse a hallucination is than a refusal, which is
    a property of the application, not the model. So sweep it.

    Utility of a policy = (correct answers served) - C x (unsupported assertions
    served), where C is the cost of one hallucination expressed in units of one
    correct answer. Ungated policy answers everything. Gated policy answers only
    items whose retrieval score clears a threshold, abstaining on the rest.

    The number that matters is the break-even C: below it, gating destroys more
    value than it protects, and the honest recommendation is "don't gate".
    """
    signal = signal or "top1_score"
    scored = [(r["signals"].get(signal, 0.0),
               r["outcome"] == CORRECT,
               r["outcome"] == UNSUPPORTED) for r in rows if r.get("signals")]
    if not scored:
        return None

    thresholds = sorted({s for s, _, _ in scored})
    # Cap the sweep so this stays cheap on large row counts.
    if len(thresholds) > 200:
        step = len(thresholds) // 200
        thresholds = thresholds[::step]

    out = {"signal": signal, "n": len(scored), "by_cost_ratio": []}
    for c in cost_ratios:
        ungated = sum(1 for _, ok, _ in scored if ok) - c * sum(1 for _, _, bad in scored if bad)

        best_util, best_t, best_kept = ungated, None, 1.0
        for t in thresholds:
            kept = [(ok, bad) for s, ok, bad in scored if s >= t]
            util = sum(1 for ok, _ in kept if ok) - c * sum(1 for _, bad in kept if bad)
            if util > best_util:
                best_util, best_t, best_kept = util, t, len(kept) / len(scored)

        out["by_cost_ratio"].append({
            "cost_ratio": c,
            "ungated_utility": round(ungated / len(scored), 4),
            "best_gated_utility": round(best_util / len(scored), 4),
            "optimal_threshold": None if best_t is None else round(best_t, 4),
            "answered_share": round(best_kept, 4),
            "gate_helps": best_t is not None,
        })

    helping = [r["cost_ratio"] for r in out["by_cost_ratio"] if r["gate_helps"]]
    out["break_even_cost_ratio"] = min(helping) if helping else None

    print(f"\nGate economics on '{signal}' "
          f"(C = cost of one hallucination in units of one correct answer)")
    print(f"  {'C':>4}{'ungated':>10}{'gated':>10}{'threshold':>12}{'answered':>11}")
    for r in out["by_cost_ratio"]:
        thr = "--" if r["optimal_threshold"] is None else f"{r['optimal_threshold']:.4f}"
        print(f"  {r['cost_ratio']:>4}{r['ungated_utility']:>10.3f}"
              f"{r['best_gated_utility']:>10.3f}{thr:>12}"
              f"{r['answered_share'] * 100:>10.0f}%")

    if out["break_even_cost_ratio"] is None:
        print("  -> No threshold beats answering everything at any tested cost ratio.")
        print("     Report this: a retrieval-score gate is not deployable here.")
    else:
        print(f"  -> Gating pays off once a hallucination costs more than "
              f"{out['break_even_cost_ratio']}x a correct answer.")
    return out


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def _thresholds(rows, signal, labels, n_points=5):
    """Operating points for a similarity gate: what a cutoff would buy in practice."""
    values = sorted(r["signals"].get(signal, 0.0) for r in rows)
    if not values:
        return []
    quantiles = [values[int(q * (len(values) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)][:n_points]

    table = []
    for threshold in quantiles:
        kept = [(r, y) for r, y in zip(rows, labels) if r["signals"].get(signal, 0.0) >= threshold]
        dropped = [(r, y) for r, y in zip(rows, labels) if r["signals"].get(signal, 0.0) < threshold]
        table.append({
            "threshold": round(threshold, 4),
            "kept_frac": round(len(kept) / len(rows), 4),
            "failure_rate_kept": round(sum(y for _, y in kept) / len(kept), 4) if kept else None,
            "failure_rate_dropped": round(sum(y for _, y in dropped) / len(dropped), 4) if dropped else None,
        })
    return table


if __name__ == "__main__":
    main()
