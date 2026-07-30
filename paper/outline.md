# Paper skeleton

Target length: 4 pages + references (workshop / SRW short paper).

**Write in this order.** Each section lists what must be true before it can be
written and what would make a reviewer reject it. Do not start a section whose
precondition is not met — writing the intro first is how a paper ends up
promising something the results do not deliver.

---

## 1. Results  *(write first)*

**Precondition:** Phases 1, 2b, 2c, 4 have run to completion on SQuAD at n=600,
`results/*.json` exist, figures generated.

**Contents**
- Table 1 — conditions × {EM, F1, abstention, hallucination, silent-failure}, each with 95% CI.
- Figure 1 — outcome composition vs. noise fraction (`fig1_noise_curve_squad.png`). **The paper's centrepiece.** H1 lives or dies here.
- Figure 2 — F1 and gold-in-context vs. K. Note where they diverge: that gap is retrieval succeeding while generation fails.
- Figure 3 — gold position at fixed K (H3, framed as replication).
- Figure 4 — mitigation trade-off scatter (H5).
- Table 2 — Phase 3 AUROC per signal + threshold operating points (H4).
- One paragraph per hypothesis: supported / not supported / underpowered. Say "not supported" where that is the answer.

**Rejected if:** any number appears without an interval; H1 is called a finding on a curve whose peak sits inside the endpoints' CIs; HotpotQA is claimed but only SQuAD was run.

---

## 2. Method

**Precondition:** Section 1 is written, so the method describes what was actually run.

**Contents**
- Pipeline: SQuAD v2 dev → 1,204 docs → 1,330 chunks (256 tok / 50 overlap) → `gemini-embedding-001` 3072-d → FAISS `IndexFlatIP` (exact, not ANN) → Gemini 2.5 Flash, temperature 0.
- Evaluation set: 600 questions, 420 answerable / 180 unanswerable, seed 20240617.
- **The design decision that needs its own paragraph:** noise is applied to the *context*, not the index. Justify it — per-item noise level is exact and known, retrieval is held identical across conditions, so conditions differ only in the named variable. A reviewer who expects index poisoning will ask.
- Outcome taxonomy (ABSTAIN / CORRECT / UNSUPPORTED / INCORRECT_GROUNDED / INVALID) and why UNSUPPORTED ≠ INCORRECT_GROUNDED. Pooling them is what makes the trivial result look like a finding.
- Judge protocol: context-grounded, never answer-only; gated to non-abstained incorrect answers; the gating is stated, not hidden.
- Statistics: percentile bootstrap CIs, paired bootstrap for all comparisons.

**Rejected if:** the method is not reproducible from the text; the judge gating is omitted; `parametric_correct_rate` is not defined before it is used.

---

## 3. Related work

**Precondition:** the papers in `RESEARCH.md` §3 have been *read*, not just listed.

**Contents** — three paragraphs, each ending in a sentence that says what this paper does that the cited work does not:
1. RAG failure benchmarks (RGB; Power of Noise) → "…both report accuracy under noise; neither decomposes failures into abstention vs. assertion."
2. Long-context and position effects (Lost in the Middle; Distracted by Irrelevant Context) → H3 is replication, stated as such.
3. Abstention and confidence (Self-RAG; Kadavath et al.; RAGAS/ARES) → Phase 3 uses retrieval-side signals available *before* generation, at zero extra model cost.

**Rejected if:** RGB or Power of Noise is missing; the noise taxonomy is presented as novel; any citation was not opened.

---

## 4. Introduction

**Precondition:** §§1–3 written, so the framing matches the evidence.

Five paragraphs: (1) RAG is deployed as the fix for hallucination; (2) but retrieval is imperfect in practice, and what matters is what the model does when retrieval fails; (3) prior work measures accuracy decay, not failure composition; (4) this paper measures composition and tests predictability — state the actual headline number from §1 here; (5) contributions, ≤3 bullets, every one traceable to a table or figure.

**Rejected if:** a contribution is claimed that no figure supports.

---

## 5. Abstract

**Precondition:** everything above is written.

Six sentences: problem → gap → what was done → **the headline number** → what it implies → the limitation that keeps it honest. No adjectives that the results do not license: no "comprehensive", no "novel framework", no "rigorous".

---

## 6. Limitations  *(required by *ACL venues)*

Straight from `RESEARCH.md` §4: self-evaluating judge, single generator family, regex abstention detection, small corpus, single sample per condition. Write it plainly. Reviewers reward this section; papers that skip it get punished for the same facts.

---

## Reproducibility checklist

- [ ] Seed and all model IDs in every results file (`manifest` block — already automated)
- [ ] `results/*.json` committed, or archived with a DOI
- [ ] Exact commands for each figure in the README
- [ ] Judge–human agreement (κ) on ~150 items reported
- [ ] Abstention-regex precision/recall on ~100 items reported
- [ ] API budget and wall-clock stated
