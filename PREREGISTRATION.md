# Preregistration

**Frozen 31 July 2026, before any experiment was run.**

The freeze is verifiable rather than asserted: this file's own commit predates every
file in `results/`. Anyone can check.

```bash
git log --format="%h %ad %s" --date=short -- PREREGISTRATION.md | tail -1
git log --format="%h %ad %s" --date=short -- results/ | tail -1
```

If the second is not strictly later than the first, this document is void.

## Why this exists

The credibility problem with a study like this is not fraud, it is drift: you run a
grid of ~27 conditions, some cell shows a pattern, and the paper gets written around
whichever pattern appeared. The framing then looks predictive when it was actually
selected after the fact. Nothing in the code can prevent that. Writing the
hypotheses down first, in a file whose timestamp is checkable, can.

## Hypotheses, as stated before data

| # | Prediction | Test | Counts as falsified if |
|---|---|---|---|
| **H1** | Silent-failure rate is **non-monotonic** in noise fraction, peaking at partial rather than total corruption | Phase 2b, `related@{25,50,75,100}` | The maximum falls at an endpoint, **or** the interior peak does not exceed both endpoints' 95% CIs |
| **H2** | `related` noise suppresses abstention **more** than `random` noise at equal fraction | Phase 2b, paired at 50% and 100% | Paired difference CI includes zero after FDR correction |
| **H3** | Gold-chunk position within a fixed context changes accuracy | Phase 2c(b), K=8, slots {0,4,7} | Paired ΔF1 (middle vs first) CI includes zero |
| **H4** | Retrieval-side signals predict silent failure above chance (AUROC > 0.6) | Phase 3, `silent_wrong` target | Best AUROC ≤ 0.6 |
| **H5** | Prompt-level abstention instructions cut hallucination only by discarding a comparable share of correct answers | Phase 4, paired Δhallucination vs Δcorrect | Δhallucination is significantly negative while Δcorrect is not, after FDR |

## Analysis decisions fixed in advance

Fixed before data because each is a defensible fork that could otherwise be chosen
to favour a result:

- **Primary outcome:** `silent_failure_rate` — share of non-correct answers that
  were assertions rather than abstentions. Not accuracy, not F1.
- **Correctness threshold:** token F1 ≥ 0.5 against gold (`config.F1_CORRECT_THRESHOLD`).
- **Hallucination:** `UNSUPPORTED` only. `INCORRECT_GROUNDED` is never pooled into it.
- **Exclusions:** `INVALID` (blocked/errored) generations, from every rate.
- **Intervals:** 95% percentile bootstrap, 2000 resamples, seed 0, clustered by
  question when repeats > 1.
- **Comparisons:** paired bootstrap on shared qids, Benjamini–Hochberg FDR at
  α=0.05 across each family of conditions-vs-control.
- **Sample:** seed 20240617, 600 SQuAD v2 dev questions, 70/30 answerable split.
- **Temperature:** 0.
- **Stopping rule:** the grid in `RESEARCH.md` §2 is run once. No conditions are
  added after seeing results without labelling them exploratory.

## What counts as exploratory

Anything not listed above. Specifically, if any of these are reported they must be
labelled post-hoc in the paper:

- results on subsets not defined here (e.g. splitting by question type);
- any noise level, K value, or mitigation not in the preregistered grid;
- any metric introduced after the first run;
- HotpotQA, if used to select the framing rather than to test whether the SQuAD
  finding replicates.

Exploratory results are legitimate and worth reporting. They are not confirmatory,
and the paper must not present them as though they were.

## Declared in advance: what happens if H1 fails

Stated now so the reframing cannot look opportunistic later. If H1 is falsified
— i.e. the silent-failure curve is monotonic, or the peak is inside the CIs — the
paper leads with Phase 3 and H4, reports H1 as a preregistered prediction that did
not hold, and targets a negative-results venue. That is the planned outcome, not a
salvage operation.

## Known deviations

Any deviation from the above is appended here with its reason and date, before the
paper is submitted.

### 2026-07-31 — retrieval backend changed from `gemini-embedding-001` to a local model

**What changed.** Document and query embeddings now come from
`sentence-transformers/all-MiniLM-L6-v2` (384-d, CPU) instead of
`gemini-embedding-001` (3072-d, API). Set `EMBED_BACKEND=gemini` to restore the
original.

**Why.** The free-tier embedding quota has a per-day cap. The first real run
exhausted it 760 chunks into a 1,330-chunk index build, and the daily embedding
allowance is not large enough to finish the index in one day. Every embedding call
also competes with the generation calls the study actually depends on.

**Was this chosen to favour a result?** No — it was forced before any outcome was
observed, and no hypothesis had been tested when the switch was made. It is logged
here rather than silently absorbed because that is the point of this document.

**Effect on the hypotheses.** None are stated in terms of retrieval quality. The
perturbations act on the context handed to the generator, and all five hypotheses
concern what the generator does *given* that context. A weaker retriever lowers
`gold_in_context_rate`, which shifts more items into the evidence-absent regime the
study is about — it does not change the direction of any prediction.

**Cost, stated plainly.** Absolute accuracy numbers are lower than they would be
with a stronger retriever and are not comparable to published SQuAD leaderboards.
Any claim about absolute performance must be scoped to this retriever.

**Benefit.** The retrieval half of the pipeline is now deterministic, free, and
reproducible without an API key, which is strictly better for a paper.
