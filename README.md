# When Retrieval Fails: A Study of Hallucination and Robustness in RAG Systems

> **Research Question:** How do noise scale, context window saturation, and zero-shot prompting strategies independently influence hallucination rates and factual recall in Retrieval-Augmented Generation (RAG) pipelines?

Retrieval-Augmented Generation (RAG) is commonly treated as a panacea for Large Language Model (LLM) hallucinations. However, enterprise systems frequently struggle with retrieval noise, unanswerable queries, and context saturation. This repository presents an empirical study evaluating the failure modes of a fundamental RAG architecture. Utilizing the SQuAD v2 dataset (containing adversarial unanswerable questions) and Google's Gemini Flash architecture, this project systematically introduces retrieval noise, varies context bounds, and measures the resulting mathematical decay in factual recall and faithfulness.

---

## Methodology & Experimental Design

The study is executed across four distinct, isolated phases to measure robust parameter relationships. The dataset consists of a 600 QA pair subset from SQuAD v2 (70% answerable, 30% unanswerable) vectorized into a local FAISS index via `models/gemini-embedding-001` (3072-dim).

- **Phase 1: Baseline Integrity (`phase1_baseline.py`)** 
  Establishes the control metrics. Evaluates standard RAG performance with $K=3$ retrieval via FAISS cosine similarity (inner product) with no added noise. 
  
- **Phase 2b: Noise Injection Degradation (`phase2b_noise_injection.py`)**
  Poisons the retrieval context. Injects up to 500 irrelevant documents into the FAISS index to measure how semantic similarity boundaries break down, and how often the LLM hallucinates over distractor text.
  
- **Phase 2c: Context Saturation (`phase2c_context_size.py`)**
  Evaluates the "Lost in the Middle" phenomenon. Scales retrieval context $K$ from 1 to 10, measuring whether expanded semantic breadth actively harms precision or faithfulness.
  
- **Phase 4: Robustness & Filtering (`phase4_fixes.py`)**
  Implements architectural mitigation. Explores stricter zero-shot prompt framing, retrieval reranking, and explicit "I don't know" classification heuristics to mathematically suppress the hallucination rate measured in Phase 2.

---

## Evaluation Metrics

To ensure academic rigor, qualitative responses are mathematically judged against strict ground-truth pairs via lexical scoring and an independent LLM-as-a-judge module.

1. **Exact Match (EM):** Binary percentage of generations that precisely match the target strings after stripping punctuation and articles.
2. **F1 Score:** Harmonic mean of precision and recall over token-level overlap. Captures partial correctness independent of strict formatting.
3. **Precision@K (P@K):** Retrieval metric evaluating if the single ground-truth document was present in the top $K$ semantic search results.
4. **Hallucination Rate:** The percentage of model generations that assert false facts specifically for *unanswerable* domain questions instead of cleanly refusing.
5. **Faithfulness Score:** An LLM-as-a-judge secondary pass (via Gemini Flash) that cross-examines the generation strictly against the provided context. Evaluates to 0 if the generated claim diverges from the provided chunks.

---

## Empirical Results

| Metric | Baseline (Phase 1) | +Noise (500 Docs) | Context $K=10$ | +Strict Prompt & Fixes |
|--------|--------------------|-------------------|----------------|------------------------|
| **Exact Match (EM)** | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* |
| **F1 Score** | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* |
| **Precision@K** | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* |
| **Hallucination Rate** | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* |
| **Faithfulness** | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* | *[ 00.0% ]* |

> *Visualizations mapping the exact decay curve of these metrics over time are available in the `/results` directory after complete execution.*

### Key Findings

- **[Insight 1 Placeholder]** (e.g., *Noise injection past 100 documents systematically caused EM to drop by X%, indicating that raw cosine similarity struggles to penalize semantically adjacent distractor paragraphs.*)
- **[Insight 2 Placeholder]** (e.g., *Increasing context window to $K=10$ ironically increased the Hallucination Rate for unanswerable questions to Y%, verifying the presence of "lost in the middle" attention phenomena.*)
- **[Insight 3 Placeholder]** (e.g., *Strict prompt heuristics alongside context reranking reduced total hallucination rates by Z%, demonstrating that generative boundaries are highly sensitive to explicit instruction over implicit context.*)

---

## Execution Guide

To reproduce this study's findings locally:

**1. Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

**2. Authentication**
Copy the environment template and insert your active Google API key (Free Tier compatible).
```bash
cp .env.example .env
# Edit .env to add GEMINI_API_KEY=AIza...
```

**3. Run the Evaluation Suite**
The execution script automatically downloads SQuAD v2, processes adversarial chunk limits, builds the FAISS databases, and tests the LLM orchestration.

```bash
# 1. Ingest Data
python data/load_squad.py

# 2. Run Phased Experiments (Sequential)
python experiments/phase1_baseline.py
python experiments/phase2b_noise_injection.py
python experiments/phase2c_context_size.py
python experiments/phase4_fixes.py

# 3. Generate Analytical Plots
python visualize_results.py
```
