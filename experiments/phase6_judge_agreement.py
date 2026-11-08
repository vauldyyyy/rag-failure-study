"""Phase 6 — is the faithfulness judge trustworthy?

The study's headline metric, the rate of unsupported assertion, is assigned by an
LLM judge. A reviewer is entitled to ask why that label should be believed, and
"we used a good prompt" is not an answer. Two checks, in increasing strength:

  1. HOW MUCH depends on the judge at all. The judge is invoked only for
     answerable items that were neither abstentions nor correct; every other label
     comes from a rule. If judge-derived labels are a small minority, the headline
     numbers are mostly rule-derived and correspondingly robust. That share should
     be reported, not assumed.

  2. INTER-JUDGE AGREEMENT. Re-judge the same items with a different model and
     report Cohen's kappa. Two independent models converging on a verdict is
     materially stronger evidence than one model asserting it. This is not a
     substitute for human labels — two LLMs can share a bias a human would not —
     but it is obtainable without annotator time, and disagreement is decisive:
     if two judges disagree often, the labels are not reportable regardless of
     what a human would have said.

Requires rows carrying `judged_context`, which records the exact evidence the
first judge ruled on. Rows collected before that field existed cannot be
re-judged, because re-deriving their context would mean replaying a stochastic
perturbation and hoping it landed identically.

    python experiments/phase6_judge_agreement.py --judge-b models/gemini-flash-lite-latest
"""

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.pipeline import save_results

JUDGE_LABELS = ("SUPPORTED", "UNSUPPORTED", "REFUSAL")


def load_rows(dataset, primary_only=False):
    """All scored rows. primary_only: untagged results files only (the primary
    generator's run), no cross-model replications and no checkpoints."""
    rows, seen = [], set()
    pattern = f"phase*_{dataset}.json" if primary_only else f"phase*_{dataset}*.json"
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("manifest", {}).get("synthetic"):
            continue
        for row in _walk(payload.get("rows")):
            key = (row.get("condition"), row.get("qid"), row.get("repeat", 0))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    for path in ([] if primary_only else sorted(glob.glob(
            os.path.join(config.RESULTS_DIR, "checkpoints", f"{dataset}_*.json")))):
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                key = (row.get("condition"), row.get("qid"), row.get("repeat", 0))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


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


def cohens_kappa(a, b, categories):
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def dependence_report(rows):
    """How much of the label set actually rests on the judge."""
    valid = [r for r in rows if r.get("outcome") != "INVALID"]
    judged = [r for r in valid if r.get("judge") in JUDGE_LABELS]
    rule_unsupported = [r for r in valid
                        if r.get("outcome") == "UNSUPPORTED"
                        and r.get("judge") == "rule:unanswerable_asserted"]
    unsupported = [r for r in valid if r.get("outcome") == "UNSUPPORTED"]
    judged_unsupported = [r for r in unsupported if r.get("judge") in JUDGE_LABELS]

    print("=" * 72)
    print("CHECK 1 — how much of the labelling depends on the judge")
    print("=" * 72)
    print(f"  valid rows                         {len(valid)}")
    print(f"  judge invoked                      {len(judged)} "
          f"({100 * len(judged) / max(len(valid), 1):.1f}%)")
    print(f"  UNSUPPORTED labels total           {len(unsupported)}")
    print(f"    from a rule (unanswerable)       {len(rule_unsupported)}")
    print(f"    from the judge                   {len(judged_unsupported)}")
    if unsupported:
        share = 100 * len(judged_unsupported) / len(unsupported)
        print(f"\n  -> {share:.0f}% of the headline metric is judge-derived; "
              f"{100 - share:.0f}% follows from a rule")
        if share < 50:
            print("     The primary outcome is majority rule-derived, so it does not")
            print("     stand or fall on judge quality. State this in Limitations.")
    return {
        "n_valid": len(valid),
        "n_judge_invoked": len(judged),
        "judge_invocation_rate": round(len(judged) / max(len(valid), 1), 4),
        "n_unsupported": len(unsupported),
        "n_unsupported_rule": len(rule_unsupported),
        "n_unsupported_judged": len(judged_unsupported),
        "judge_derived_share_of_headline": (
            round(len(judged_unsupported) / len(unsupported), 4) if unsupported else None),
    }


def agreement_report(rows, judge_b_model, limit):
    """Re-judge with a second model and compare."""
    from utils.metrics import judge_faithfulness

    eligible = [r for r in rows
                if r.get("judge") in JUDGE_LABELS and r.get("judged_context")]

    print("\n" + "=" * 72)
    print("CHECK 2 — inter-judge agreement")
    print("=" * 72)
    if not eligible:
        print("  No rows carry `judged_context`.")
        print("  That field was added after the current results were collected, so")
        print("  those verdicts cannot be re-audited: reconstructing their context")
        print("  would mean replaying a stochastic perturbation and assuming it")
        print("  landed identically. Rows collected from now on are re-judgeable.")
        print("\n  This is itself a reportable finding: until now the study could not")
        print("  reproduce its own judge decisions.")
        return None

    sample = eligible[:limit]
    print(f"  Re-judging {len(sample)} of {len(eligible)} eligible rows "
          f"with {judge_b_model}\n")

    # Restore from config, not the environment: JUDGE_MODEL is usually unset in
    # the environment, so the old code never restored it and recorded judge B
    # as judge A in the report.
    original = config.JUDGE_MODEL
    config.JUDGE_MODEL = judge_b_model
    import utils.metrics as M
    M._judge_model = None  # force the client to rebuild against the new model

    a_labels, b_labels, disagreements = [], [], []
    for i, row in enumerate(sample):
        verdict_b = judge_faithfulness(
            row["question"], row["judged_context"], row["prediction"])
        if verdict_b not in JUDGE_LABELS:
            continue
        a_labels.append(row["judge"])
        b_labels.append(verdict_b)
        if row["judge"] != verdict_b:
            disagreements.append({
                "qid": row.get("qid"), "condition": row.get("condition"),
                "judge_a": row["judge"], "judge_b": verdict_b,
                "prediction": (row.get("prediction") or "")[:160],
            })
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(sample)}")

    config.JUDGE_MODEL = original

    if not a_labels:
        print("  No comparable verdicts returned.")
        return None

    raw = sum(x == y for x, y in zip(a_labels, b_labels)) / len(a_labels)
    kappa = cohens_kappa(a_labels, b_labels, list(JUDGE_LABELS))

    print(f"\n  n compared      {len(a_labels)}")
    print(f"  raw agreement   {raw:.3f}")
    print(f"  Cohen's kappa   {kappa:.3f}" if kappa is not None else "  kappa n/a")
    print(f"  disagreements   {len(disagreements)}")
    print("\n  " + _verdict(kappa))

    confusion = {}
    for a, b in zip(a_labels, b_labels):
        confusion.setdefault(a, {}).setdefault(b, 0)
        confusion[a][b] += 1

    return {
        "judge_a": config.JUDGE_MODEL,
        "judge_b": judge_b_model,
        "n": len(a_labels),
        "raw_agreement": round(raw, 4),
        "cohens_kappa": None if kappa is None else round(kappa, 4),
        "confusion": confusion,
        "disagreements": disagreements[:25],
    }


def _verdict(k):
    if k is None:
        return ""
    if k >= 0.8:
        return ("-> Strong agreement. Two independent models converge; the labels "
                "are reportable, with human validation still preferable.")
    if k >= 0.6:
        return ("-> Substantial agreement. Report kappa alongside every "
                "judge-derived number rather than presenting the labels as exact.")
    return ("-> Weak agreement. Do not lead with judge-derived hallucination rates. "
            "Lead with the rule-derived unanswerable rate and report this kappa as "
            "the reason.")


def main():
    p = argparse.ArgumentParser(description="Phase 6: judge dependence and agreement")
    p.add_argument("--dataset", default="squad", choices=["squad", "hotpot"])
    p.add_argument("--judge-b", default="models/gemini-flash-lite-latest",
                   help="Second judge; must differ from the primary judge")
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--primary-only", action="store_true",
                   help="Re-judge only rows from the primary run's results files")
    p.add_argument("--agreement-only", action="store_true",
                   help="Keep the existing dependence block and replace only the "
                        "agreement block. The dependence numbers are reported in the "
                        "paper for the primary run; recomputing them over every "
                        "results file (cross-model runs included) would change them.")
    args = p.parse_args()

    rows = load_rows(args.dataset, primary_only=args.primary_only)
    if not rows:
        sys.exit(f"No results for '{args.dataset}'.")

    print(f"{len(rows)} scored rows\n")
    dependence = None
    if args.agreement_only:
        existing = os.path.join(config.RESULTS_DIR,
                                f"phase6_judge_agreement_{args.dataset}.json")
        with open(existing, "r", encoding="utf-8") as f:
            dependence = json.load(f).get("dependence")
    report = {
        "dependence": dependence if dependence is not None else dependence_report(rows),
        "agreement": agreement_report(rows, args.judge_b, args.limit),
        "note": ("Inter-judge agreement bounds label reliability from one side only: "
                 "two LLM judges can share a bias no human would. Disagreement is "
                 "decisive against the labels; agreement is supportive, not proof."),
    }
    save_results(f"phase6_judge_agreement_{args.dataset}.json", report)


if __name__ == "__main__":
    main()
