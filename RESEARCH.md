# Research plan

> **Target: UncertaiNLP @ EMNLP 2026 — short paper (4 pages) — deadline 7 August 2026.**
> Backup: Insights (Negative Results in NLP) @ EMNLP 2026, Budapest, 22–29 Oct.

Working document. Everything here is a commitment the code has to keep, or a gap
that is still open. It is written to be edited, not admired.

---

## 1. Research question

> **When retrieval degrades, does a RAG system fail loudly or silently — and can the
> failure be predicted from retrieval-side signals before the generator is called?**

Two clauses, both testable, both measurable with the pipeline in this repo:

* **Loudly or silently.** For every failed answer, the system either abstained
  ("I cannot answer from the context") or asserted something unsupported. The
  ratio between those two is the dependent variable. Accuracy is not the object
  of study; the *composition of failures* is.
* **Predictable in advance.** Similarity scores and their shape exist before the
  LLM is called. If they separate failures from successes, an abstention gate is
  buildable. If they do not, that is a negative result worth stating.

### What this deliberately is not

"How does RAG performance degrade under noise" — because the answer is known
(it goes down), already published, and unreviewable.

### Hypotheses

| # | Hypothesis | Where tested | Falsified if |
|---|---|---|---|
| **H1** | Silent-failure rate is **non-monotonic** in noise fraction: it peaks at *partial* corruption and falls at total corruption, because a fully-useless context triggers abstention while a partly-plausible one licenses a confident guess. | Phase 2b | The curve is monotonic, or the interior peak is inside the CI of the endpoints |
| **H2** | **Related** noise (topically adjacent, wrong) suppresses abstention more than **random** noise at equal fraction. | Phase 2b | Abstention rates overlap within CI |
| **H3** | Gold-chunk **position** inside a fixed context changes accuracy (lost-in-the-middle) independently of context size. | Phase 2c(b) | Paired ΔF1 across positions is not significant |
| **H4** | Retrieval scores predict silent failure better than chance (AUROC > 0.6). | Phase 3 | AUROC ≈ 0.5 → negative result, still reportable |
| **H5** | Prompt-level abstention instructions reduce hallucination **only** by trading away correct answers roughly one-for-one; a relevance filter does better. | Phase 4 | Strict prompting cuts hallucination without a comparable accuracy cost |

H1 and H4 are the ones worth a paper. H2, H3, H5 are supporting or replication.

---

## 2. Experiment matrix

**Failure categories** (independent variable, applied to the context handed to the
generator — never to the index, so the per-item noise level is exact and known):

| Category | Implementation | `utils/noise.py` |
|---|---|---|
| Irrelevant retrieval | context slots replaced by uniformly sampled corpus chunks | `RANDOM` |
| Partial retrieval | slots replaced by lower-ranked real retrieval results (adjacent, wrong) | `RELATED` |
| Contradictory retrieval | gold answer string in the context substituted with a false one | `COUNTERFACTUAL` |
| No answer in context | every gold-document chunk removed, question still answerable | `DROP_GOLD` |

Each row records `treatment_applied` and `n_perturbed`, because `COUNTERFACTUAL`
silently no-ops when the gold string is absent from the retrieved chunk and
`DROP_GOLD` no-ops when gold was never retrieved. An undiluted condition and a
condition that fired on 40% of items are different experiments.

**Datasets** (two, so no finding rests on one benchmark's quirks):

| Dataset | Why | Gives |
|---|---|---|
| SQuAD v2 dev | `is_impossible` questions are a clean abstention probe: gold paragraph present, topically perfect, contains no answer | 600 QA (420 answerable / 180 unanswerable), 1,204 documents → 1,330 chunks |
| HotpotQA dev (distractor) | multi-hop, so partial retrieval is the *native* condition; ships 8 human-selected hard distractors per question | 600 QA, different domain mix, no unanswerable split |

**Grid:**

| Phase | Varies | Held fixed | Conditions |
|---|---|---|---|
| 1 | — | K=3, clean | 1 |
| 2b | noise type × fraction | K=4 | 1 clean + 9 |
| 2c(a) | K ∈ {1,2,3,4,5,6,8,10} | clean retrieval | 8 |
| 2c(b) | gold slot ∈ {0,4,7} | K=8, identical context contents | 3 |
| 4 | mitigation | related@50%, K=4 | 5 |
| 3 | — (re-analysis of rows above) | — | 0 API calls |

**Metrics.** EM and token-F1 (official SQuAD definitions, max over annotator
answers, multiset overlap), P@K, recall@K, plus the outcome taxonomy that carries
the actual argument:

```
ABSTAIN             declined to answer
CORRECT             F1 >= 0.5 against gold
UNSUPPORTED         asserted something the retrieved context does not support   <- hallucination
INCORRECT_GROUNDED  wrong, but traceable to the context                         <- retrieval's fault
INVALID             blocked/errored generation; excluded from all rates
```

Derived: `silent_failure_rate` = share of non-correct answers that were assertions
rather than abstentions. `parametric_correct_rate` = correct *without* the gold
document in context, i.e. answered from weights — this contaminates every accuracy
number in every retrieval ablation and is reported separately.

**Statistics.** Every rate carries a 95% percentile-bootstrap CI. Every
between-condition claim is a paired bootstrap on the same questions, with a
two-sided p. Repeated samples of the same question are averaged within the question
before resampling, so `--repeats` can never shrink an interval. No claim is made
from a point estimate.

---

## 3. Related work — positioning

**All 12 entries verified 31 Jul 2026** against ACL Anthology, PMLR, AAAI and
NeurIPS proceedings, and arXiv — title, authors, venue, year and pages.
`paper/refs.bib` carries the source for each. Any future entry marked
`NOT YET RE-VERIFIED` will fail `verify_paper.py` if cited.

### The two that overlap directly

**Chen et al., *Benchmarking LLMs in RAG* (RGB), AAAI 2024, 38(16):17754–17762.**
Verified. RGB measures four abilities: noise robustness, **negative rejection**,
information integration, and counterfactual robustness. Their finding: LLMs show
some noise robustness but "struggle significantly in negative rejection,
information integration, and dealing with false information."

> This is closer to the present work than first assumed. *Negative rejection* is
> abstention. The taxonomy is theirs and must be cited as theirs. What remains
> distinct: RGB scores rejection as a **capability at a fixed noise ratio**;
> this study traces how the abstain/assert split moves as corruption varies
> **continuously**, per item, at a known dose. Frame the contribution as the
> dose–response curve, never as the taxonomy.

**Cuconasu et al., *The Power of Noise*, SIGIR 2024, pp. 719–729 (arXiv:2401.14887).**
Verified. Finding: adding **random** documents to the prompt *improves* accuracy by
up to 35%, while related-but-non-answer-bearing documents degrade it.

> H2 predicts the same direction. If it replicates, it is a **replication on a
> different generator**, and the paper must say so in those words. The part they do
> not decompose is what the model does when it is wrong — abstain or assert. That
> decomposition is the contribution, not the direction of the effect.

### Supporting

| Work | Status | Role |
|---|---|---|
| Liu et al., *Lost in the Middle*, TACL 2024 (arXiv:2307.03172) | Verified | H3 is an explicit small-scale replication. Say "replication" in the results text |
| Yoran et al., *Making RALMs Robust to Irrelevant Context*, ICLR 2024 (arXiv:2310.01558) | Verified | Their mitigation is training-based; scope Phase 4's claim to inference-time defences only |
| Es et al., *RAGAS*, EACL 2024 demo, pp. 150–158 | Verified | Justify the narrower custom judge rather than reusing theirs |
| Saad-Falcon et al., *ARES*, NAACL 2024, pp. 338–354 (arXiv:2311.09476) | Verified | Same |
| Rajpurkar et al., *SQuAD 2.0*, ACL 2018 (short), pp. 784–789 | Verified | Required dataset citation; also the source of the abstention framing |
| Yang et al., *HotpotQA*, EMNLP 2018, pp. 2369–2380 | Verified | Required dataset citation |
| Lewis et al., *RAG*, NeurIPS 2020, 33:9459–9474 | Verified | Opening citation |
| Asai et al., *Self-RAG*, ICLR 2024 (arXiv:2310.11511) | Verified | Phase 3 is the cheap baseline their method should beat |
| Kadavath et al., *LMs (Mostly) Know What They Know*, arXiv:2207.05221 | Verified | Never formally published — cite as a preprint. They use model internals; Phase 3 uses retrieval-side signals, which is the distinction to draw |
| Shi et al., *Distracted by Irrelevant Context*, ICML 2023, PMLR 202:31210–31227 | Verified | Non-RAG version of the same effect |
| Longpre et al., *Entity-Based Knowledge Conflicts*, EMNLP 2021, pp. 7052–7063 | Verified | Prior art for the `COUNTERFACTUAL` condition |

### Honest read on novelty

After RGB and Power of Noise, what survives:

1. **Failure composition as a dose–response curve.** Abstention vs unsupported
   assertion measured across a continuous, per-item-exact corruption fraction (H1).
2. **Pre-generation predictability (Phase 3).** A retrieval-score abstention gate
   with reported operating points — the free baseline that Self-RAG-style methods
   should have to beat. **This is the strongest remaining claim** and it costs no
   extra API calls.
3. **`parametric_correct_rate` as a reported confound.** Most RAG ablations do not
   separate "retrieval worked" from "the model already knew it".

If H1 comes out flat, contribution 1 collapses and the paper becomes a
**negative-result / reproducibility** paper built on 2 and 3 — which is exactly what
the Insights workshop exists for. Decide *after* the data, not before.

---

## 4. Threats to validity

| Threat | Status |
|---|---|
| **Judge shares a checkpoint with the generator** — faithfulness scores are self-evaluation | **Harness built, labels not yet collected.** `JUDGE_SELF_EVAL` is stamped into every results file; `JUDGE_MODEL` is overridable; `annotate.py` draws a seeded stratified sample, hides the automatic label from the annotator, and reports Cohen's κ. **Do this — it is the cheapest credibility available.** |
| Abstention detected by rule | **Harness built.** `annotate.py --task abstention` reports precision/recall/F1 of the rule against human labels. Hedged assertions are already excluded by construction and tested |
| Single generator family | **Addressable.** `--model` threads through every phase; one extra run gives cross-model evidence |
| Single sample per condition | **Addressable.** `--repeats N --temperature T`; `stats.collapse_repeats` averages within question first so intervals stay honest |
| Perturbation silently no-ops | **Fixed.** `treatment_applied_rate` and `mean_slots_perturbed` recorded per condition |
| Corpus is 1,330 chunks vs "enterprise knowledge base" framing | **Open.** Enterprise framing removed from the README; keep it out of the abstract |
| SQuAD paragraphs mostly fit one chunk, so chunking is near-trivial | Known. Make no claims about chunking strategy from this setup |
| `parametric_correct_rate` confound | Measured and reported |
| Pipeline bugs discovered mid-run | **Fixed.** `--dry-run` executes every phase against a deterministic stub with no API key; CI runs it on every push |
| Prose drifting from data | **Fixed.** Every number in the paper is generated by `paper/make_tables.py`; `verify_paper.py` fails the build on hardcoded numbers, undefined macros, missing figures, unverified citations, or synthetic results |
| `google-generativeai` SDK is end-of-life | Open. Pinned to 0.8.5 so runs reproduce; migrate to `google-genai` before camera-ready |

---

## 5. Write order (do not deviate)

Results → Method → Related work → Introduction → Abstract → Title.

The abstract is written last because it must describe what the data showed, not
what the plan hoped for. `paper/main.tex` is drafted in that order with each
section's precondition in a comment; acceptance criteria in
[`paper/outline.md`](paper/outline.md).

---

## 6. Submission target — locked

**UncertaiNLP (3rd Workshop on Uncertainty-Aware NLP) @ EMNLP 2026.**

| | |
|---|---|
| **Deadline** | **7 August 2026** (direct submission) |
| Track | Short paper, up to 4 pages — explicitly for "ongoing or preliminary work" |
| Scope fit | Lists **selective prediction** and **calibration** as topics. Phase 3 *is* selective prediction |
| Camera-ready | +1 page for reviewer feedback |
| Site | <https://uncertainlp.github.io/> — re-check dates there, they were TBD for ARR commitment |

Why this venue over an SRW: the workshop's stated scope names the exact
contribution that survives the related-work analysis (§3). Phase 3 is the paper;
H1 is supporting evidence.

**Backup if the data is not ready or H1 is flat:** Insights (Negative Results in
NLP) @ EMNLP 2026, Budapest, 22–29 October. A monotonic H1 plus a chance-level H4
is a publishable negative result there. Check their CFP page for the deadline.

### Backward plan from 7 August

| By | Do |
|---|---|
| **31 Jul** (today) | Add API key. Launch `python run_all.py --limit 150`. **This is the critical path — everything else waits on it** |
| 1 Aug | Read the H1 curve from the pilot. If flat, switch the framing to Phase 3 + negative result now, not later |
| 1–2 Aug | Launch the full 600 run if quota allows; otherwise commit to n=150 and say so |
| 2–3 Aug | `annotate.py sample/label/score` for both tasks (~3 h of human work). Verify the SQuAD/HotpotQA/Lewis/Self-RAG/Kadavath/Shi/Longpre citations |
| 3–4 Aug | Write Results, then Method (`paper/main.tex` sections are already drafted) |
| 5 Aug | Related work, Introduction |
| 6 Aug | Abstract, Limitations, `python verify_paper.py --strict` until clean |
| 7 Aug | Submit |

**Quota risk:** the free tier has a daily request cap as well as 15 RPM. 27
conditions × 600 items ≈ 16k generations will not fit in a few days of free-tier
quota. Check your limit before committing to the full run; **n=150 with honest
confidence intervals is a perfectly good short paper** and fits the "preliminary
work" track. Do not let the full run become the reason nothing is submitted.

---

## 7. Open work, in order

1. **Run the pilot.** `python run_all.py --limit 150`. Everything else is blocked on this.
2. Read the H1 curve. Re-frame if flat.
3. `annotate.py` for judge κ and abstention precision/recall.
4. ~~Verify the remaining citations~~ — done 31 Jul 2026, all 12 verified.
5. Full run and/or HotpotQA if quota allows.
6. Write in the order in §5. `python verify_paper.py --strict` must exit 0 before submitting.
