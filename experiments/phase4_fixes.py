"""Phase 4 — mitigations, evaluated where the failure actually happens.

Two corrections to the original design:

1. Mitigations are applied under the condition Phase 2b identified as actually
   dangerous -- COUNTERFACTUAL at 50%, where the context states something false --
   not on clean retrieval and not on the noise the generator already survives.
   Under absent or diluted evidence the model abstains and needs no defence; under
   falsified evidence it repeats the falsehood with no rise in abstention at all.
   A defence measured where the pipeline was not failing shows nothing.

2. The outcome is a trade-off, not a single number. Any prompt that pushes a model
   toward abstention will reduce hallucination — and will also refuse questions it
   could have answered. So each mitigation reports the pair
   (fall in hallucination rate, fall in correct rate). A mitigation only counts as
   a win if the first clearly exceeds the second.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import noise as noise_mod
from utils.metrics import ABSTAIN, CORRECT, UNSUPPORTED
from utils.pipeline import (announce_budget, quota_guard, build_retrieval_cache, common_args,
                            load_dataset, run_condition, run_manifest, save_results)
from utils.rerank import make_reranker
from utils.stats import apply_fdr, paired_diff_by_qid, rate_with_ci_clustered

K = 4
# Repointed from RELATED to COUNTERFACTUAL after Phase 2b. Mitigations were
# originally tested under adjacent-but-wrong evidence, chosen before the data
# showed where the danger actually is: falsified evidence, which produces a
# tenfold rise in grounded error with no rise in abstention. Testing prompt-level
# defences under noise the generator already handles safely answers nothing.
# The live question is whether any inference-time defence detects a context that
# lies -- and the taxonomy predicts it cannot, because nothing looks uncertain.
NOISE_TYPE = noise_mod.COUNTERFACTUAL
NOISE_FRAC = 0.5

STRICT_PROMPT = (
    "You answer strictly from the provided context. "
    "If the context does not explicitly state the answer, reply with exactly: "
    "I cannot answer based on the context. "
    "Do not infer, deduce, or use prior knowledge. Do not guess."
)

CITED_PROMPT = (
    "You answer strictly from the provided context. "
    "Give the answer, then on a new line write 'Source: Context N' naming the "
    "context that states it. If no context states the answer, reply with exactly: "
    "I cannot answer based on the context."
)

MITIGATIONS = [
    ("none", None, None),
    ("strict_prompt", STRICT_PROMPT, None),
    ("cite_source", CITED_PROMPT, None),
    ("bm25_rerank", None, make_reranker(keep=2)),
    ("strict+rerank", STRICT_PROMPT, make_reranker(keep=2)),
]


def main():
    args = common_args("Phase 4: mitigations under degraded retrieval")
    qa_pairs, corpus, store, cache_path = load_dataset(
        args.dataset, limit=args.limit, rebuild_index=args.rebuild_index
    )
    retrieval = build_retrieval_cache(qa_pairs, store, cache_path)

    announce_budget(len(MITIGATIONS), len(qa_pairs), use_judge=not args.no_judge,
                    repeats=args.repeats)
    print(f"--- Phase 4: mitigations under {NOISE_TYPE}@{int(NOISE_FRAC * 100)}%, "
          f"K={K}, on {args.dataset} ---")

    conditions, all_rows = {}, {}
    for name, prompt, rerank in MITIGATIONS:
        rows, summary = run_condition(
            name, qa_pairs, retrieval, corpus, k=K,
            noise_type=NOISE_TYPE, noise_frac=NOISE_FRAC,
            system_prompt=prompt, rerank=rerank,
            use_judge=not args.no_judge, model_name=args.model,
            repeats=args.repeats, temperature=args.temperature,
            dataset=args.dataset,
        )
        valid = [r for r in rows if r["outcome"] != "INVALID"]
        summary["ci"] = {
            "hallucination_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == UNSUPPORTED),
            "abstention_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == ABSTAIN),
            "correct_rate": rate_with_ci_clustered(valid, lambda r: r["outcome"] == CORRECT),
        }
        conditions[name], all_rows[name] = summary, rows
        print(f"  {name:15s} halluc={summary.get('hallucination_rate')} "
              f"abstain={summary.get('abstention_rate')} F1={summary.get('F1')}")

        # Persist after every mitigation. On free-tier quota this phase spans days;
        # writing only at the end means a partial run yields no results file and so
        # no figure, despite the rows being on disk.
        save_results(f"phase4_fixes_{args.dataset}.json", {
            "manifest": run_manifest(args.dataset, qa_pairs, {
                "k": K, "noise_type": NOISE_TYPE, "noise_frac": NOISE_FRAC,
                "mitigations": [m[0] for m in MITIGATIONS],
                "mitigations_collected": len(conditions),
                "complete": len(conditions) == len(MITIGATIONS),
            }),
            "conditions": conditions, "rows": all_rows,
        })

    control = all_rows["none"]
    print("\nTrade-off vs no mitigation (paired, 95% CI):")
    print(f"  {'mitigation':16s}{'d halluc':>22}{'d correct':>22}   verdict")
    for name, _, _ in MITIGATIONS[1:]:
        rows = all_rows[name]
        d_hall = _paired(control, rows, lambda r: 1.0 if r["outcome"] == UNSUPPORTED else 0.0)
        d_corr = _paired(control, rows, lambda r: 1.0 if r["outcome"] == CORRECT else 0.0)
        conditions[name]["vs_none"] = {"hallucination_rate": d_hall, "correct_rate": d_corr}

        conditions[name]["verdict"] = _verdict(d_hall, d_corr)

    # Four mitigations against one control: correct before declaring a winner.
    for metric in ("hallucination_rate", "correct_rate"):
        apply_fdr({name: (conditions[name].get("vs_none") or {}).get(metric)
                   for name, _, _ in MITIGATIONS[1:]})

    for name, _, _ in MITIGATIONS[1:]:
        vs = conditions[name]["vs_none"]
        d_hall, d_corr = vs["hallucination_rate"], vs["correct_rate"]
        # A mitigation that only looked significant before correction is not one.
        if d_hall and not d_hall.get("significant_fdr") and conditions[name]["verdict"] == "net win":
            conditions[name]["verdict"] = "not significant after FDR correction"
        q = "" if not d_hall else f"  q={d_hall.get('q_value')}"
        print(f"  {name:16s}{_fmt(d_hall):>22}{_fmt(d_corr):>22}   "
              f"{conditions[name]['verdict']}{q}")

    save_results(f"phase4_fixes_{args.dataset}.json", {
        "manifest": run_manifest(args.dataset, qa_pairs, {
            "k": K, "noise_type": NOISE_TYPE, "noise_frac": NOISE_FRAC,
            "mitigations": [m[0] for m in MITIGATIONS],
        }),
        "conditions": conditions,
        "rows": all_rows,
    })


def _paired(control_rows, rows, fn):
    return paired_diff_by_qid(control_rows, rows, fn)


def _fmt(d):
    if not d:
        return "n/a"
    return f"{d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]"


def _verdict(d_hall, d_corr):
    """A mitigation wins only if it removes more hallucination than accuracy."""
    if not d_hall or not d_corr:
        return "n/a"
    if d_hall["hi"] >= 0:
        return "no significant reduction in hallucination"
    if d_corr["lo"] <= 0 and abs(d_corr["diff"]) >= abs(d_hall["diff"]):
        return "trades away as much accuracy as it saves"
    return "net win"


if __name__ == "__main__":
    quota_guard(main)()
