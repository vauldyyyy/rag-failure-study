# When Retrieval Fails

**An empirical study of how RAG systems behave when retrieval degrades — and whether the failure can be seen coming.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/FAISS-IndexFlatIP-FF6F00?style=flat-square" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Data-SQuAD%20v2%20%2B%20HotpotQA-blueviolet?style=flat-square" alt="Datasets"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT"/>
</p>

<p align="center">
  <b>Status: pipeline complete, tested and dry-run verified end-to-end · experiments not yet run · no results to report</b><br/>
  <sub>Target: UncertaiNLP @ EMNLP 2026 short paper — deadline 7 August 2026</sub>
</p>

---

## Research question

> **When retrieval degrades, does a RAG system fail loudly or silently — and can the
> failure be predicted from retrieval-side signals before the generator is called?**

The first clause matters because a RAG system that says *"I cannot answer from the
context"* is safe, and one that produces a fluent wrong answer from the same broken
context is not — yet both count identically as an accuracy drop. This study measures
the **composition of failures**, not the accuracy drop.

The second clause matters because similarity scores exist before the LLM is called.
If they separate failures from successes, an abstention gate is buildable for free.
If they do not, that is a negative result worth publishing.

Plan, hypotheses, verified prior-work positioning, threats to validity and the
dated submission plan: [`RESEARCH.md`](RESEARCH.md).
Paper skeleton: [`paper/outline.md`](paper/outline.md). Draft: [`paper/main.tex`](paper/main.tex).

---

## Results

**There are none yet.** The pipeline is implemented, unit-tested, and has been
executed end-to-end against a stub model to verify every code path — but no
experiment has been run against the API. This section will hold Table 1 and
Figures 1–4 once `run_all.py` has completed.

Nothing in this repository reports a number the pipeline did not produce, and that
is enforced rather than promised:

- every number in the paper is generated from `results/*.json` by `paper/make_tables.py`;
- `verify_paper.py` fails the build on hardcoded numbers, undefined macros, missing
  figures, uncited or unverified references, template placeholders, or results
  produced in dry-run mode;
- CI asserts that the guard *rejects* synthetic results, so the guard itself cannot
  silently break.

---

## What is measured

Every generation lands in exactly one outcome:

| Outcome | Meaning |
|---|---|
| `ABSTAIN` | Declined to answer — a **safe** failure |
| `CORRECT` | Token F1 ≥ 0.5 against gold |
| `UNSUPPORTED` | Asserted something the retrieved context does not support — **hallucination** |
| `INCORRECT_GROUNDED` | Wrong, but traceable to the context — a **retrieval** failure, not a generation one |
| `INVALID` | Blocked or errored generation; excluded from every rate |

Pooling `UNSUPPORTED` and `INCORRECT_GROUNDED` is what turns "RAG fails when
retrieval is bad" into something that looks like a finding. They are kept apart.

Two derived quantities carry most of the argument:

- **`silent_failure_rate`** — of the answers that were not correct, what share were
  assertions rather than abstentions.
- **`parametric_correct_rate`** — answered correctly *without* the gold document in
  context, i.e. from the model's weights. This inflates the baseline of every
  retrieval ablation and is almost never reported.

Every rate carries a 95% bootstrap CI; every between-condition comparison is a
paired bootstrap over the same questions. Repeated samples of a question are
averaged within that question before resampling, so `--repeats` cannot manufacture
significance.

---

## Experimental design

Perturbations are applied to the **context handed to the generator**, not to the
FAISS index. Index poisoning makes the noise level a property of the whole run, so
you cannot say what noise level any individual answer was produced under.
Context-level perturbation fixes the fraction exactly, per item, and holds retrieval
byte-identical across conditions.

| Condition | Failure mode it isolates |
|---|---|
| `RANDOM` | Irrelevant retrieval — topically unrelated chunks |
| `RELATED` | Partial retrieval — adjacent, plausible, wrong (real lower-ranked results) |
| `COUNTERFACTUAL` | Contradictory retrieval — the context asserts a false answer |
| `DROP_GOLD` | No answer in context — evidence absent, question still answerable |

Each condition records `treatment_applied_rate`, because `COUNTERFACTUAL` no-ops
when the gold string is absent from the chunk and `DROP_GOLD` no-ops when gold was
never retrieved. A condition that fired on 40% of items is a different experiment
from one that fired on all of them.

| Phase | Varies | Fixed | Cells |
|---|---|---|---|
| 1 | — | K=3, clean | 1 |
| 2b | noise type × fraction | K=4 | 10 |
| 2c(a) | K ∈ {1,2,3,4,5,6,8,10} | clean | 8 |
| 2c(b) | gold slot ∈ {0,4,7} | K=8, identical contents | 3 |
| 4 | mitigation | related@50%, K=4 | 5 |
| 3 | — re-analysis, 0 API calls | — | — |

Data: **SQuAD v2 dev** — 1,204 documents → 1,330 chunks, 600 questions (420
answerable / 180 unanswerable, seed 20240617). The unanswerable split is the point:
gold paragraph present, topically perfect, no answer in it. Second domain:
**HotpotQA dev (distractor)**, multi-hop with human-selected hard distractors.

---

## Getting started

```bash
git clone https://github.com/vauldyyyy/rag-failure-study.git
cd rag-failure-study
python -m venv venv && venv\Scripts\activate      # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python data/load_squad.py                         # corpus + eval set — no API key needed
```

**Retrieval runs locally by default** (`sentence-transformers/all-MiniLM-L6-v2` on
CPU) — no API key, no quota, ~3 minutes to build the index, and reproducible by
anyone. Only generation and judging need a key. Set `EMBED_BACKEND=gemini` in
`.env` to use `gemini-embedding-001` instead, but note that 1,330 chunks exceeds
the free tier's *daily* embedding cap, so the index cannot be built in one day.

### Try it with no API key

```bash
python run_all.py --dry-run --limit 60
```

Runs every phase against a deterministic stub model. Output goes to
`results_dryrun/`, figures are watermarked **SYNTHETIC**, and every results file is
stamped `"synthetic": true`. It verifies the plumbing, not the findings — a dry-run
curve that looks like it supports a hypothesis is an artefact of the stub.

### Run for real

```bash
cp .env.example .env      # then add a key from https://aistudio.google.com
python run_all.py --limit 150     # pilot — read the Phase 2b curve before committing quota
python run_all.py                 # full 600-item run
```

Or phase by phase:

```bash
python experiments/phase1_baseline.py
python experiments/phase2b_noise_injection.py
python experiments/phase2c_context_size.py
python experiments/phase4_fixes.py
python experiments/phase3_predictability.py    # analysis only, no API calls
python visualize_results.py
```

Useful flags on every phase: `--limit N`, `--no-judge`, `--model <id>` (cross-model
generality), `--repeats N --temperature T` (sampling variance), `--dataset hotpot`.

**Budget.** Every script prints its API cost before starting. The full SQuAD grid is
~27 conditions × 600 items ≈ 16k generations plus judge calls — at the free tier's
15 RPM and daily cap that is days, not hours. Check your quota before committing;
**n=150 with honest confidence intervals is a perfectly good short paper.** Query
embeddings and retrieval are cached to disk, so a crashed phase never re-pays for
retrieval, and an invalid key fails in seconds rather than hanging on backoff.

### Check the automatic labels against a human

```bash
python annotate.py sample --task judge --n 150   # seeded, stratified, fixed before labelling
python annotate.py label  --task judge           # automatic label hidden to avoid anchoring
python annotate.py score  --task judge           # Cohen's kappa
python annotate.py sample --task abstention --n 100
python annotate.py score  --task abstention      # precision/recall of the abstention rule
```

This is the cheapest credibility the project can buy, and it is the one thing the
code cannot do for you. The judge currently shares a checkpoint with the generator;
without κ, every judge-based number is self-evaluation.

### Build the paper

```bash
python paper/make_tables.py --dataset squad      # numbers -> paper/generated/
python verify_paper.py --strict                  # must exit 0 before submitting
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Tests

```bash
python tests/test_offline.py
```

49 checks over scoring, abstention detection, the perturbations, the statistics
(including that repeated sampling cannot shrink an interval), and the dry-run stub.
No API key required. Runs in CI on every push, along with a full dry-run pipeline.

### Console

```bash
streamlit run app.py
```

Reproduce any condition interactively; browse real scored rows from `results/`.

---

## Project structure

```
rag-failure-study/
├── RESEARCH.md              # question, hypotheses, verified related work, threats, dated plan
├── paper/
│   ├── main.tex             # 4-page draft; every number comes from a generated macro
│   ├── refs.bib             # citations, each marked verified or not
│   ├── make_tables.py       # results/*.json -> LaTeX tables + inline macros
│   └── outline.md           # write-order skeleton with acceptance criteria
├── verify_paper.py          # submission gate: no unbacked number reaches a PDF
├── annotate.py              # human labelling + Cohen's kappa / rule precision-recall
├── run_all.py               # every phase in dependency order
├── app.py                   # Streamlit console
├── visualize_results.py     # figures with bootstrap error bars
├── config.py                # models, seeds, thresholds, paths, dry-run switch
│
├── data/
│   ├── load_squad.py        # full-dev corpus + stratified 70/30 answerable split
│   └── load_hotpot.py       # second domain, same schema
│
├── experiments/
│   ├── phase1_baseline.py           # control
│   ├── phase2b_noise_injection.py   # noise type x fraction  (H1, H2)
│   ├── phase2c_context_size.py      # K sweep + gold position (H3)
│   ├── phase3_predictability.py     # pre-generation failure prediction (H4)
│   └── phase4_fixes.py              # mitigations + accuracy/safety trade-off (H5)
│
├── utils/
│   ├── pipeline.py          # the single shared run loop every phase uses
│   ├── noise.py             # the four perturbations
│   ├── metrics.py           # outcome taxonomy, SQuAD scoring, context-grounded judge
│   ├── stats.py             # bootstrap CIs, paired tests, AUROC, repeat clustering
│   ├── rerank.py            # BM25 relevance filter (mitigation)
│   ├── stub_llm.py          # deterministic dry-run model
│   ├── embedder.py          # Gemini embeddings + disk cache
│   ├── llm.py               # generation wrapper
│   ├── vector_store.py      # FAISS IndexFlatIP
│   └── chunker.py           # token-aware splitter
│
├── tests/test_offline.py    # 49 checks, runs without an API key
└── results/                 # JSON with provenance manifest + figures — tracked
```

---

## Known limitations

Stated up front because they bound what any result here can claim. Each one now has
either a fix or a harness; none is hand-waved.

- **The judge shares a checkpoint with the generator** — faithfulness verdicts are
  self-evaluation. Every results file records `judge_self_eval: true`. Set
  `JUDGE_MODEL` to a different model, and run `annotate.py` to report κ.
- **One generator family.** `--model` threads through every phase; one extra run
  gives cross-model evidence.
- **Abstention is detected by rule.** Hedged assertions are excluded by
  construction and tested, but the rule's precision/recall needs human labels —
  `annotate.py --task abstention`.
- **1,330 chunks is not an enterprise knowledge base.** Claims are scoped to this
  corpus size.
- **One sample per condition** at temperature 0. `--repeats N --temperature T`
  addresses this; intervals stay clustered by question.
- `google-generativeai` is end-of-life; pinned for reproducibility, migration to
  `google-genai` is tracked in [`RESEARCH.md`](RESEARCH.md).

---

## License

MIT — see [LICENSE](LICENSE).
