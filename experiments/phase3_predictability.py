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

    interp = (
        "AUROC near 0.5 means retrieval scores carry no usable pre-generation signal; "
        "a system cannot gate on them and must fall back to post-hoc verification. "
        "AUROC above ~0.7 means an abstention gate is viable and the threshold table "
        "gives its operating points."
    )
    report["interpretation"] = interp
    print(interp)

    save_results(f"phase3_predictability_{args.dataset}.json", report)


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
