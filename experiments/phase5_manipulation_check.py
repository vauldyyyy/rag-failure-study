"""Phase 5 — manipulation check: does the synthetic perturbation reproduce real failure?

The central vulnerability of this study is that its independent variable is
artificial. We break retrieval on purpose, then report how the generator responds.
A reviewer is entitled to ask whether any of it describes retrieval that breaks on
its own, and "the perturbations are principled" is not an answer.

It is answerable from data already collected, at no API cost, because the clean
condition contains naturally failed retrievals: items where the gold document
simply was not in the top-K. Those are real failures produced by a real retriever
on real questions, with no intervention.

Two checks:

  1. CONVERGENT VALIDITY. Compare the outcome composition of naturally
     evidence-free items (clean condition, gold absent) against synthetically
     evidence-free items (drop_gold, related@100). If the generator behaves the
     same way, the manipulation reproduces the phenomenon and the experimental
     conditions inherit that credibility. If it does not, that is a real limitation
     and the paper must say so rather than wait to be told.

  2. OBSERVATIONAL DOSE-RESPONSE. Within the clean condition alone, bin items by
     retrieval score and plot silent-failure rate per bin. This is the correlational
     shadow of the experimental curve, computed on entirely unmanipulated data. If
     the observational and experimental curves point the same way, the causal claim
     is much harder to dismiss as an artefact of how the noise was constructed.

Neither check can be run before the experiment; both are free once it exists.
Costs zero API calls.
"""

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.metrics import ABSTAIN, CORRECT, INCORRECT_GROUNDED, UNSUPPORTED
from utils.pipeline import save_results
from utils.stats import bootstrap_ci, rate_with_ci_clustered


def load_rows(dataset="squad"):
    """Every scored row, from finished results files or live checkpoints."""
    rows, seen = [], set()

    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, f"phase*_{dataset}.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("manifest", {}).get("synthetic"):
            continue
        for row in _walk(payload.get("rows")):
            key = (row.get("condition"), row.get("qid"), row.get("repeat", 0))
            if key not in seen:
                seen.add(key)
                rows.append(row)

    # Checkpoints cover conditions whose phase has not finished writing yet.
    ckpt_dir = os.path.join(config.RESULTS_DIR, "checkpoints")
    for path in sorted(glob.glob(os.path.join(ckpt_dir, f"{dataset}_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                key = (row.get("condition"), row.get("qid"), row.get("repeat", 0))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

    return [r for r in rows if r.get("outcome") != "INVALID"]


def _walk(node):
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "outcome" in item:
                yield item
            else:
                yield from _walk(item)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)


def composition(rows):
    """Outcome shares with intervals, plus the derived silent-failure rate."""
    if not rows:
        return None
    out = {"n": len(rows)}
    for label in (CORRECT, ABSTAIN, UNSUPPORTED, INCORRECT_GROUNDED):
        out[label] = rate_with_ci_clustered(rows, lambda r, L=label: r["outcome"] == L)
    failed = [r for r in rows if r["outcome"] != CORRECT]
    out["silent_failure_rate"] = (
        rate_with_ci_clustered(failed, lambda r: r["outcome"] != ABSTAIN) if failed else None
    )
    return out


def _fmt(ci):
    if not ci or ci.get("mean") is None:
        return "  --  "
    return f"{ci['mean']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]"


def overlaps(a, b):
    """Do two bootstrap intervals overlap? A deliberately weak agreement test.

    Non-overlap is strong evidence of a real difference; overlap is only weak
    evidence of equivalence, especially at this n. Reported as such, never as
    'the two are the same'.
    """
    if not a or not b or a.get("mean") is None or b.get("mean") is None:
        return None
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])


# --------------------------------------------------------------------------- #

def convergent_validity(rows):
    """Natural evidence-free items vs synthetically evidence-free items."""
    clean = [r for r in rows if r.get("condition") in ("clean", "baseline")]
    natural_fail = [r for r in clean if r.get("gold_in_context") is False]
    natural_ok = [r for r in clean if r.get("gold_in_context") is True]

    synthetic = [r for r in rows
                 if r.get("condition") in ("drop_gold@100", "related@100")
                 and r.get("gold_in_context") is False]

    print("=" * 74)
    print("CHECK 1 — convergent validity: natural vs synthetic evidence-free items")
    print("=" * 74)
    if not natural_fail:
        print("No naturally failed retrievals in the clean condition yet.")
        return None
    if not synthetic:
        print(f"Natural evidence-free items: n={len(natural_fail)}")
        print("No synthetic evidence-free condition has run yet "
              "(needs drop_gold@100 or related@100). Re-run after Phase 2b completes.")
        return {"natural_fail": composition(natural_fail),
                "natural_ok": composition(natural_ok), "synthetic": None}

    nat, syn = composition(natural_fail), composition(synthetic)
    print(f"\n{'outcome':<22}{'natural (n=%d)' % nat['n']:<26}{'synthetic (n=%d)' % syn['n']:<26}agree?")
    print("-" * 74)
    verdicts = {}
    for label in (CORRECT, ABSTAIN, UNSUPPORTED, INCORRECT_GROUNDED, "silent_failure_rate"):
        ov = overlaps(nat.get(label), syn.get(label))
        verdicts[label] = ov
        mark = "--" if ov is None else ("yes" if ov else "NO")
        print(f"{label:<22}{_fmt(nat.get(label)):<26}{_fmt(syn.get(label)):<26}{mark}")

    disagree = [k for k, v in verdicts.items() if v is False]
    print()
    if not disagree:
        print("-> Intervals overlap on every outcome. The manipulation reproduces what")
        print("   the retriever does when it fails unaided, within the resolution of")
        print("   this sample. State it that way: overlap at n of this size is weak")
        print("   evidence of equivalence, not proof.")
    else:
        print(f"-> Divergence on: {', '.join(disagree)}.")
        print("   The synthetic condition does NOT reproduce natural failure on these")
        print("   outcomes. This belongs in Limitations, stated plainly, and it bounds")
        print("   how far the experimental results generalise.")

    return {"natural_fail": nat, "natural_ok": composition(natural_ok),
            "synthetic": syn, "interval_overlap": verdicts}


def observational_dose_response(rows, n_bins=4):
    """Silent failure vs retrieval score, on unmanipulated data only."""
    clean = [r for r in rows
             if r.get("condition") in ("clean", "baseline") and r.get("signals")]
    print("\n" + "=" * 74)
    print("CHECK 2 — observational dose-response (clean condition, no manipulation)")
    print("=" * 74)
    if len(clean) < n_bins * 10:
        print(f"Only {len(clean)} clean rows; too few to bin.")
        return None

    scored = sorted(clean, key=lambda r: r["signals"].get("mean_score", 0.0))
    size = len(scored) // n_bins
    bins = []
    print(f"\n{'retrieval score':<22}{'n':<6}{'gold in ctx':<14}{'silent failure'}")
    print("-" * 74)
    for i in range(n_bins):
        chunk = scored[i * size:(i + 1) * size] if i < n_bins - 1 else scored[i * size:]
        if not chunk:
            continue
        lo = chunk[0]["signals"].get("mean_score", 0.0)
        hi = chunk[-1]["signals"].get("mean_score", 0.0)
        gold = bootstrap_ci([1.0 if r.get("gold_in_context") else 0.0 for r in chunk])
        failed = [r for r in chunk if r["outcome"] != CORRECT]
        silent = (rate_with_ci_clustered(failed, lambda r: r["outcome"] != ABSTAIN)
                  if failed else None)
        bins.append({"score_lo": round(lo, 4), "score_hi": round(hi, 4),
                     "n": len(chunk), "gold_in_context": gold,
                     "silent_failure_rate": silent})
        print(f"{f'{lo:.3f}-{hi:.3f}':<22}{len(chunk):<6}"
              f"{_fmt(gold):<14}{_fmt(silent)}")

    usable = [b for b in bins if b["silent_failure_rate"]
              and b["silent_failure_rate"]["mean"] is not None]
    if len(usable) >= 3:
        vals = [b["silent_failure_rate"]["mean"] for b in usable]
        peak = vals.index(max(vals))
        interior = 0 < peak < len(vals) - 1
        print()
        if interior:
            print(f"-> Silent failure peaks in an interior score bin ({peak + 1} of "
                  f"{len(vals)}), on data with no manipulation at all.")
            print("   This is the observational shadow of H1. If the experimental")
            print("   curve peaks too, the finding is much harder to dismiss as an")
            print("   artefact of how the noise was built.")
        else:
            print(f"-> Silent failure is monotone across score bins (peak at bin "
                  f"{peak + 1} of {len(vals)}); no interior peak observationally.")
    return bins


def main():
    p = argparse.ArgumentParser(description="Phase 5: manipulation check (no API calls)")
    p.add_argument("--dataset", default="squad", choices=["squad", "hotpot"])
    args = p.parse_args()

    rows = load_rows(args.dataset)
    if not rows:
        sys.exit(f"No results for '{args.dataset}'. Run Phase 1 and 2b first.")

    conditions = sorted({r.get("condition") for r in rows})
    print(f"{len(rows)} scored rows across {len(conditions)} conditions: "
          f"{', '.join(conditions)}\n")

    report = {
        "n_rows": len(rows),
        "conditions": conditions,
        "convergent_validity": convergent_validity(rows),
        "observational_dose_response": observational_dose_response(rows),
        "note": ("Interval overlap is weak evidence of equivalence, not proof. "
                 "Non-overlap is strong evidence of difference."),
    }
    save_results(f"phase5_manipulation_check_{args.dataset}.json", report)


if __name__ == "__main__":
    main()
