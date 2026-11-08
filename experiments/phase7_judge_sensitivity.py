"""Phase 7: does the headline depend on the LLM judge? (0 API calls)

An independent judge (phase 6) disagreed with the primary judge on every
UNSUPPORTED label it re-checked, so judge-assigned UNSUPPORTED labels cannot be
taken at face value. This phase recomputes the two headline comparisons against
the clean condition under the two defensible readings:

  rule_only    Hallucination counts only rule-derived labels (an assertion on an
               unanswerable question, where SQuAD guarantees no answer exists).
               Judge-assigned UNSUPPORTED labels are ignored.
  reclassified Judge-assigned UNSUPPORTED labels are treated as grounded error,
               the reading phase 6's disagreements point to.

It also reports label dependence computed over the complete grid of each run.

    python experiments/phase7_judge_sensitivity.py
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.stats import apply_fdr, paired_diff_by_qid

RUNS = {"primary": "phase2b_noise_squad.json", "gemma": "phase2b_noise_squad_gemma.json"}


def is_rule(r):
    return str(r.get("judge") or "").startswith("rule:")


def is_unsupported(r):
    return r["outcome"] == "UNSUPPORTED"


def dependence(rows):
    valid = [r for r in rows if r["outcome"] != "INVALID"]
    unsup = [r for r in valid if is_unsupported(r)]
    judged = [r for r in valid if r.get("judge") in ("SUPPORTED", "UNSUPPORTED", "REFUSAL")]
    rule = sum(1 for r in unsup if is_rule(r))
    return {
        "n_rows": len(rows), "n_valid": len(valid), "n_invalid": len(rows) - len(valid),
        "n_judge_invoked": len(judged),
        "judge_invocation_pct": round(100 * len(judged) / len(valid), 1),
        "n_unsupported": len(unsup), "n_unsupported_rule": rule,
        "n_unsupported_judged": len(unsup) - rule,
        "rule_derived_pct": round(100 * rule / len(unsup), 1) if unsup else None,
    }


def analyse(data):
    rows = data["rows"]
    control = rows["clean"]
    names = [n for n in rows if n != "clean"]

    def family(fn):
        comps = {n: paired_diff_by_qid(control, rows[n], fn) for n in names}
        return apply_fdr(comps)

    rule_only = family(lambda r: is_unsupported(r) and is_rule(r))
    grounded_plus = family(lambda r: r["outcome"] == "INCORRECT_GROUNDED"
                           or (is_unsupported(r) and not is_rule(r)))

    def summarise(comps):
        up = [n for n in names if comps[n].get("significant_fdr") and comps[n]["diff"] > 0]
        down = [n for n in names if comps[n].get("significant_fdr") and comps[n]["diff"] < 0]
        nulls = [n for n in names if n not in up and n not in down]
        bound = max((max(abs(comps[n]["lo"]), abs(comps[n]["hi"])) for n in nulls), default=None)
        return {"n": len(names), "up": up, "down": down, "n_null": len(nulls),
                "null_bound_pts": None if bound is None else round(100 * bound, 1),
                "max_upper_pts": round(100 * max(comps[n]["hi"] for n in names), 1)}

    all_rows = [r for v in rows.values() for r in v]
    return {
        "generator": data["manifest"].get("llm_model"),
        "dependence": dependence(all_rows),
        "rule_only_hallucination": {"summary": summarise(rule_only), "by_condition": rule_only},
        "reclassified_grounded": {"summary": summarise(grounded_plus),
                                  "counterfactual@50": grounded_plus.get("counterfactual@50"),
                                  "by_condition": grounded_plus},
    }


def missingness(data, reference):
    """How much do questions the generator never finished matter? (0 API calls)

    Some questions send a reasoning model into a loop that outlasts any timeout, so
    their rows are INVALID. Re-run the headline comparisons on the questions valid in
    all ten conditions, so differential loss between conditions cannot produce them,
    and compare excluded and retained questions on the fully observed reference run.
    """
    rows = data["rows"]
    all_q = {r["qid"] for r in rows["clean"]}
    bad = {r["qid"] for v in rows.values() for r in v if r["outcome"] == "INVALID"}
    keep = all_q - bad
    sub = {n: [r for r in v if r["qid"] in keep] for n, v in rows.items()}
    names = [n for n in sub if n != "clean"]

    def family(fn):
        return apply_fdr({n: paired_diff_by_qid(sub["clean"], sub[n], fn) for n in names})

    hallu = family(is_unsupported)
    outcomes = {"correct": "CORRECT", "abstain": "ABSTAIN", "grounded": "INCORRECT_GROUNDED",
                "unsupported": "UNSUPPORTED"}
    cf = {k: family(lambda r, o=o: r["outcome"] == o)["counterfactual@50"]
          for k, o in outcomes.items()}

    def rate(rows_, qids, outcome):
        hit = [r for r in rows_ if r["qid"] in qids]
        return round(100 * sum(r["outcome"] == outcome for r in hit) / len(hit), 1)

    def unans(qids):
        hit = [r for r in rows["clean"] if r["qid"] in qids]
        return round(100 * sum(not r.get("is_answerable", True) for r in hit) / len(hit), 1)

    ref = reference["rows"]["clean"]
    per_q = [sum(r["outcome"] == "INVALID" and r["qid"] == q for v in rows.values() for r in v)
             for q in all_q]
    return {
        "n_questions": len(all_q), "n_complete": len(keep), "n_excluded": len(bad),
        "n_excluded_5plus": sum(c >= 5 for c in per_q),
        "unanswerable_pct": {"excluded": unans(bad), "complete": unans(keep)},
        "reference_clean": {o.lower(): {"excluded": rate(ref, bad, o), "complete": rate(ref, keep, o)}
                            for o in outcomes.values()},
        "hallucination": {
            "up": [n for n in names if hallu[n].get("significant_fdr") and hallu[n]["diff"] > 0],
            "down": [n for n in names if hallu[n].get("significant_fdr") and hallu[n]["diff"] < 0],
            "max_upper_pts": round(100 * max(hallu[n]["hi"] for n in names), 1)},
        "counterfactual@50": cf,
    }


def main():
    out = {"note": ("Sensitivity of the headline comparisons to judge-assigned labels. "
                    "No API calls; recomputed from per-item rows.")}
    for tag, name in RUNS.items():
        path = os.path.join(config.RESULTS_DIR, name)
        if not os.path.exists(path):
            print(f"{tag}: {name} missing, skipped")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out[tag] = analyse(data)
        s = out[tag]
        print(f"\n{tag} ({s['generator']})\n  dependence: {s['dependence']}")
        print(f"  rule-only hallucination: {s['rule_only_hallucination']['summary']}")
        cf = s["reclassified_grounded"]["counterfactual@50"]
        print(f"  grounded+judged at counterfactual@50: diff {cf['diff']:+.3f} "
              f"[{cf['lo']:+.3f}, {cf['hi']:+.3f}] q={cf.get('q_value')}")
    if "gemma" in out:
        with open(os.path.join(config.RESULTS_DIR, RUNS["gemma"]), encoding="utf-8") as f:
            gemma = json.load(f)
        with open(os.path.join(config.RESULTS_DIR, RUNS["primary"]), encoding="utf-8") as f:
            primary = json.load(f)
        out["gemma"]["missingness"] = missingness(gemma, primary)
        print("\nmissingness:", json.dumps(out["gemma"]["missingness"], default=str)[:1800])
    target = os.path.join(config.RESULTS_DIR, "phase7_judge_sensitivity_squad.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {target}")


if __name__ == "__main__":
    main()
