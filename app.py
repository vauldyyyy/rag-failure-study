"""Interactive console.

Note on the Failure Case Viewer: it reads real rows out of results/*.json. An
earlier version of this file shipped hand-written example failures with invented
model outputs and invented noise levels, presented as specimens from the Phase 2
run. Nothing in this app displays a number the pipeline did not produce.
"""

import glob
import json
import os
import random

import streamlit as st

import config
from utils import noise as noise_mod
from utils.embedder import get_embedding
from utils.llm import generate_answer
from utils.metrics import is_abstention, judge_faithfulness
from utils.vector_store import VectorStore

st.set_page_config(page_title="RAG Failure Console", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource
def load_corpus(dataset):
    corpus_path, _, _, _ = config.dataset_paths(dataset)
    if not os.path.exists(corpus_path):
        return []
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_vector_store(dataset):
    _, _, index_path, _ = config.dataset_paths(dataset)
    if not os.path.exists(index_path):
        return None
    return VectorStore().load(index_path)


@st.cache_data
def load_result_rows():
    """Every scored row this study has written, flattened."""
    rows = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "phase*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in _walk(payload.get("rows")):
            row["_source"] = os.path.basename(path)
            rows.append(row)
    return rows


def _walk(node):
    if node is None:
        return
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "outcome" in item:
                yield item
            else:
                yield from _walk(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)


dataset = st.sidebar.selectbox("Dataset", ["squad", "hotpot"])
corpus = load_corpus(dataset)
store = load_vector_store(dataset)

st.title("When Retrieval Fails — exploration console")
st.caption("Reproduce any condition from the study interactively. "
           "Live generations call the Gemini API and cost quota.")

tab1, tab2, tab3 = st.tabs(["Query playground", "Observed failures", "Results"])

# --------------------------------------------------------------------------- #
with tab1:
    if store is None:
        st.error(f"No index for '{dataset}'. Run `python data/load_{dataset}.py` "
                 f"then any experiment phase to build it.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            query = st.text_input("Question", "In what country is Normandy located?")
            k_val = st.slider("Context size K", 1, 10, config.TOP_K)
            noise_type = st.selectbox(
                "Perturbation",
                [noise_mod.CLEAN, noise_mod.RELATED, noise_mod.RANDOM,
                 noise_mod.COUNTERFACTUAL, noise_mod.DROP_GOLD],
            )
            noise_frac = st.slider("Share of context slots perturbed", 0.0, 1.0, 0.5, 0.25,
                                   disabled=noise_type == noise_mod.CLEAN)
            strict = st.checkbox("Strict abstention prompt")
            run_btn = st.button("Run", type="primary")

        with col2:
            if run_btn and query:
                with st.spinner("Retrieving..."):
                    emb = get_embedding(query, task_type="RETRIEVAL_QUERY")
                    ranked = store.search(emb, top_k=config.RETRIEVAL_DEPTH)

                # No gold document is known for a free-text query, so gold-relative
                # perturbations degrade to "perturb the top slots".
                pseudo_qa = {"question": query, "answers": [], "gold_doc_id": None}
                context = noise_mod.apply_noise(
                    ranked, pseudo_qa, k_val,
                    noise_type=noise_type,
                    noise_frac=0.0 if noise_type == noise_mod.CLEAN else noise_frac,
                    corpus=corpus, rng=random.Random(config.SEED),
                )
                if noise_type == noise_mod.DROP_GOLD:
                    st.info("drop_gold needs a labelled gold document; on a free-text "
                            "query it behaves as clean retrieval.")

                prompt = (
                    "You answer strictly from the provided context. If the context does "
                    "not explicitly state the answer, reply with exactly: "
                    "I cannot answer based on the context."
                ) if strict else None

                with st.spinner("Generating..."):
                    gen = generate_answer(query, context, system_prompt=prompt)

                st.subheader("Answer")
                if gen["status"] != "ok":
                    st.error(f"Generation {gen['status']}: {gen.get('detail', '')}")
                else:
                    st.info(gen["text"])
                    abstained = is_abstention(gen["text"])
                    if abstained:
                        st.success("Abstained — safe failure")
                    else:
                        with st.spinner("Judging against the shown context..."):
                            verdict = judge_faithfulness(query, context, gen["text"])
                        if verdict == "UNSUPPORTED":
                            st.error("UNSUPPORTED — asserts something the context does not support")
                        elif verdict == "SUPPORTED":
                            st.success("SUPPORTED by the context shown below")
                        else:
                            st.warning(f"Judge returned: {verdict}")
                        if config.JUDGE_SELF_EVAL:
                            st.caption("Judge and generator are the same model — "
                                       "this verdict is self-evaluation.")

                st.subheader(f"Context actually sent to the model ({len(context)} chunks)")
                for i, c in enumerate(context):
                    tag = c.get("perturbed")
                    label = f"Context {i + 1} — sim {c.get('score', 0):.4f}"
                    if tag:
                        label += f"  [injected: {tag}]"
                    with st.expander(label, expanded=i == 0):
                        st.write(c["text"])

# --------------------------------------------------------------------------- #
with tab2:
    rows = load_result_rows()
    if not rows:
        st.info("No result files yet. Run the experiment phases to populate this tab.")
    else:
        outcomes = sorted({r["outcome"] for r in rows})
        pick = st.selectbox("Outcome", outcomes,
                            index=outcomes.index("UNSUPPORTED") if "UNSUPPORTED" in outcomes else 0)
        subset = [r for r in rows if r["outcome"] == pick]
        st.caption(f"{len(subset)} of {len(rows)} scored generations")

        conditions = sorted({r.get("condition", "?") for r in subset})
        cond = st.selectbox("Condition", ["(all)"] + conditions)
        if cond != "(all)":
            subset = [r for r in subset if r.get("condition") == cond]

        for r in subset[:40]:
            with st.expander(f"[{r.get('condition')}] {r['question'][:90]}"):
                st.markdown(f"**Gold:** `{r.get('gold_answers') or 'UNANSWERABLE'}`")
                st.markdown(f"**Model:** {r.get('prediction')}")
                st.markdown(
                    f"**F1** {r.get('f1')} · **gold in context** {r.get('gold_in_context')} "
                    f"· **judge** {r.get('judge')} · **source** `{r.get('_source')}`"
                )

# --------------------------------------------------------------------------- #
with tab3:
    figs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "fig*.png")))
    if not figs:
        st.info("No figures yet. Run `python visualize_results.py` after the phases.")
    for path in figs:
        st.image(path, caption=os.path.basename(path))

    summaries = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "phase*.json")))
    for path in summaries:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        with st.expander(os.path.basename(path)):
            st.json({k: v for k, v in payload.items() if k != "rows"})
