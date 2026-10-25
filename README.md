# When Retrieval Fails: Hallucination and Robustness in RAG

> **Research question.** How do retrieval noise, conflicting evidence, context size, the position of
> the relevant passage, and abstention prompting change hallucination and factual recall in a
> retrieval-augmented generation (RAG) pipeline?

Status: **pipeline complete and tested; full model runs pending.** Results tables are generated
from the run outputs (`python run.py report`), never typed in by hand.

## Design

Data come from the **SQuAD v2.0 dev set**, which mixes answerable questions with adversarial
*unanswerable* ones (written against a real paragraph so they look answerable from it).

| Part | Construction |
|---|---|
| Base corpus | Every paragraph of 12 seeded-random Wikipedia articles (397 passages). Same-article paragraphs act as on-topic hard negatives. |
| Questions | 300 answerable + 150 unanswerable, sampled from those articles (seed 13). |
| Noise pool | 807 paragraphs from the 23 *other* articles, in a fixed shuffled order; noise level *n* adds the first *n*, so levels are nested. |
| Counterfactuals | For each answerable question, a copy of its source paragraph with the gold answer replaced by a same-type answer from another question. |
| Retrieval | `gemini-embedding-001` (768-d, L2-normalised) + FAISS inner-product search. |
| Generation | `gemini-2.5-flash`, temperature 0, thinking off, short-answer prompt. Any OpenAI-compatible model can be swapped in for cross-model runs. |

### Experiments

| ID | Manipulation | Questions |
|---|---|---|
| E1 | Baseline: K=3, no noise, default prompt | all |
| E2 | 0 / 100 / 250 / 500 off-topic distractors in the index | all |
| E2b | Counterfactual paragraphs in the index (conflicting evidence) | answerable |
| E3 | K = 1, 2, 3, 5, 7, 10 retrieved passages | all |
| E4 | 10 passages, gold paragraph forced to position 1 / 3 / 5 / 7 / 10 ("lost in the middle") | answerable |
| E5 | Prompt open / default / strict × noise 0 / 500 | all |

### Metrics

- **EM / F1** on answerable questions, official SQuAD normalisation, max over gold answers.
- **Hallucination rate**: share of *unanswerable* questions the model answered instead of abstaining.
- **Over-refusal**: share of *answerable* questions the model refused. E5 measures the trade-off between the two.
- **Retrieval hit@K and MRR** of the source paragraph.
- **Misled rate** (E2b): the model returned the planted counterfactual answer when that paragraph was retrieved.
- **Faithfulness** (`--judge`): an LLM judge, shown the same context, labels each non-abstained answer SUPPORTED / UNSUPPORTED.
- Every metric reports a **95% bootstrap CI**; condition contrasts use a **paired bootstrap** on the same questions.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY

python run.py build-data        # data/processed/dataset.json (already committed)
python run.py plan              # API calls needed (~9.5k upper bound, ~9.5M input tokens)
python run.py run E1 --limit 20 # smoke test on 20 questions
python run.py run all --judge   # full study; safe to stop and resume
python run.py report            # results/<model>/RESULTS.md + figures/
```

- Every embedding and response is cached in `cache/cache.sqlite`, keyed by its exact inputs. An
  interrupted run resumes, and prompts shared between experiments are paid for once.
- On the free tier set `RAG_RPM` to your requests-per-minute limit; a full run then spans several
  days of quota. On a paid key, `--workers 4` parallelises requests.
- `--gen openai` with `OPENAI_BASE_URL` (Groq, OpenRouter, Ollama) runs the same experiments on a
  second model; results go to a separate `results/<model>/` folder for comparison.
- `python -m pytest` runs the suite offline using a deterministic mock backend
  (`--gen mock --embed mock`). Mock numbers are not results.

## Layout

```
config.py              all settings (seed, sample sizes, conditions, models)
run.py                 CLI: build-data | plan | run | report
ragstudy/data.py       dataset, distractor pool, counterfactuals
ragstudy/study.py      experiment definitions, retrieval, runner
ragstudy/backends.py   Gemini, OpenAI-compatible, offline mock
ragstudy/metrics.py    SQuAD scoring, abstention detection, bootstrap statistics
ragstudy/prompts.py    answer and judge prompts
analysis/report.py     tables, paired tests, figures
docs/paper_outline.md  structure for the write-up
tests/                 unit + end-to-end tests
```

## Known limitations

- One embedding model; distractors are off-topic Wikipedia paragraphs (E2b covers on-topic conflict).
- Abstention is detected by pattern; spot-check `results/*/E5/open_*.jsonl`, where the model was never told the token.
- The faithfulness judge is the same model family as the generator; report judge agreement on a hand-labelled sample before relying on it.
