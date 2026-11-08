# Cross-model replication: run notes (8 Nov 2026)

What was run, and the operational details that do not belong in the paper but
matter for anyone reproducing it.

## Configuration

| | Primary run | Replication |
|---|---|---|
| Generator | `gemini-3.5-flash-lite` | `gemma-4-26b-a4b-it` (open weights, different family) |
| Judge | `gemini-3.1-flash-lite` | `gemini-3.1-flash-lite` (same) |
| Questions | 150 SQuAD v2 (114 / 36) | the same 150, same order (qids checked) |
| Retrieval | local MiniLM + FAISS, K = 4 | identical; gold-in-context matches the primary run item by item |
| Perturbations | 9 conditions + clean, seed 20240617 | identical |
| Prompt | system instruction + user turn | same text; Gemma takes no system instruction, so it is prepended to the user turn |
| Temperature | 0 | 0 |

Results: `results/phase2b_noise_squad_gemma.json` (tag `gemma`).

## Why Gemma 4 26B

- Gemma 3 27B, planned first, has been retired from the Gemini API (404).
- Gemma 4 31B was overloaded at run time: mean 25 s per call and roughly one
  call in four failing with 503 "high demand". The 26B mixture-of-experts model
  answered every probe at about 7 s.

## Things that had to be fixed for the run to be valid

1. **Reasoning leaked into answers.** Gemma 4 returns its reasoning as parts
   flagged `thought: true`; the pinned `google-generativeai` SDK drops the flag,
   so `response.text` was 800+ characters of reasoning instead of a span. Gemma
   is now called over REST and only non-thought parts are kept
   (`utils/llm.py::_rest_generate`).
2. **Provenance.** `--model` reached the generator but not `run_manifest`, so a
   replication would have been labelled as the primary model. Fixed in
   `utils/pipeline.py::common_args`.
3. **Transient server errors dropped items.** 500/503/504 and read timeouts were
   retried five times quickly and the item then became INVALID, shrinking n.
   They now get up to ten extra retries with short backoff and no quota
   penalty (`utils/ratelimit.py::is_transient_server_error`).

## Rate limits (free tier, read from AI Studio on 8 Nov 2026)

| Model | RPM | TPM (input) | RPD |
|---|---|---|---|
| gemma-4-26b-a4b-it | 30 | 16K | 14.4K |
| gemini-3.1-flash-lite (judge) | 15 | 250K | 500 |

A Gemma prompt is about 700 input tokens, so 16K TPM allows about 22 calls per
minute across all processes. The grid was collected with
`experiments/phase2b_worker.py`, one process per condition, `LLM_RPM=2` each
(20 per minute in total), then aggregated by `phase2b_noise_injection.py`, which
finds every condition complete in its checkpoint and only computes statistics.

Observed at run time: 7–31 s per Gemma call (250–1,300 reasoning tokens) and
about one call in five failing transiently (timeout, 503 or 500) before retry.

## Items that never finish

Some prompts send Gemma 4 into a reasoning loop that outlasts any timeout, and
at temperature 0 they do so on every attempt (question 7, "What is a connection
identifier", hangs every time in the clean condition). The API rejects
`thinkingBudget` and `thinkingLevel` for this model, so reasoning cannot be
capped. Such items now get one retry after a 45 s read timeout and are then
recorded as `gen_status: error` (INVALID: excluded from every rate, counted in
`n_invalid`, reported in the paper). Before this, each one consumed fifteen
attempts, about fifteen minutes. Checkpoints are written every five items.

## Outcome

| | |
|---|---|
| Generations | 1,500 (10 conditions x 150 questions) |
| Invalid | 162 (10.8%) across 46 questions; 20 questions fail in 5 or more conditions and account for 126 rows |
| Hallucination vs clean | up in 0 of 9 conditions, 3 significant decreases, 6 null; no 95% CI above +2.5 points |
| 50% counterfactual | correct -19.5 points, grounded error +18.7 (3.1% to 22.0%), abstention +0.8 (q = 0.846) |
| Complete-case (104 questions) | correct -19.2, grounded +18.3, abstention +1.0; hallucination up in 0 of 9 |

Invalid rows were retried in a repair pass (`LLM_RPM=2`, `GEN_TIMEOUT=90`) after the
main run, which reduced them from 184 to 162; the rest are the reasoning loops above
or persistent server errors. Loss is not random: in the fully observed primary run,
the 46 affected questions have a clean unsupported rate of 8.7% against 2.9% for the
other 104. `experiments/phase7_judge_sensitivity.py` computes all of this.
