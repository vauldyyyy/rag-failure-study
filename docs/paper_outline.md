# Paper outline (target: arXiv preprint, then an NLP/IR student or workshop venue)

Working title: *When Retrieval Fails: Noise, Conflict and Position Effects on Hallucination in RAG*

## 1. Introduction
- RAG is deployed to reduce hallucination, but retrieval is imperfect.
- Questions: (1) Does off-topic retrieval noise hurt answers or only retrieval? (2) Does the model
  follow conflicting retrieved evidence? (3) Does more context help or hurt? (4) Does the gold
  passage's position matter for a current model? (5) What does abstention prompting buy, and at
  what cost in refusals?
- Contributions: a controlled, reproducible protocol on SQuAD v2 with paired statistics; the
  hallucination / over-refusal trade-off curve; released code and cached outputs.

## 2. Related work (verify each citation before writing)
- Retrieval-augmented generation — Lewis et al., 2020.
- SQuAD 2.0 unanswerable questions — Rajpurkar, Jia & Liang, 2018.
- Lost in the middle — Liu et al., 2023 (TACL 2024).
- Irrelevant context distracts LLMs — Shi et al., 2023 (ICML).
- Noise in retrieved documents — Cuconasu et al., 2024 ("The Power of Noise", SIGIR).
- RAG robustness benchmark (RGB) — Chen et al., 2024 (AAAI).
- Knowledge conflicts — Xie et al., 2024 (ICLR); ClashEval — Wu et al., 2024.

## 3. Method
Dataset construction, retrieval setup, prompts (quote all three in an appendix), metrics,
bootstrap CIs and paired tests. Table: experiment grid (copy from README).

## 4. Results — one subsection per experiment
Each: figure from `results/<model>/figures/`, headline number with CI, paired test.
- E2 noise: retrieval hit@3 vs F1 vs hallucination.
- E2b conflict: misled rate; does it depend on whether the gold paragraph is also present?
- E3 context size: does F1 saturate; does hallucination on unanswerables rise with K?
- E4 position: U-shape or flat?
- E5 prompting: hallucination vs over-refusal scatter; SQuAD-v2 EM as the combined score.

## 5. Analysis
- Error taxonomy on ~100 sampled failures (hand-labelled): wrong passage, right passage wrong
  span, copied a near-miss span for an unanswerable, followed counterfactual, refused wrongly.
- Judge validation: agreement of the LLM judge with your labels on that sample.
- Second model (`--gen openai`) to show whether effects generalise.

## 6. Limitations and 7. Conclusion

## Checklist before posting
- [ ] Full run on the primary model with `--judge`
- [ ] Second model run
- [ ] Hand-labelled error sample + judge agreement
- [ ] Every number in the text traced to `RESULTS.md`
- [ ] Release `results/` and the cache-free code; add a licence
