import streamlit as st
import json
import os
import random
import config
from utils.vector_store import VectorStore
from utils.embedder import get_embedding
from utils.llm import generate_answer
from utils.metrics import is_hallucinated

# Page config
st.set_page_config(page_title="RAG Failure Console", layout="wide", initial_sidebar_state="expanded")

# Initialize custom dark-mode aesthetic styling
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        border-left: 4px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Cache Loaders
@st.cache_resource
def load_corpus():
    if not os.path.exists(config.CORPUS_PATH): return []
    with open(config.CORPUS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_vector_store():
    store = VectorStore(dimension=3072)
    if os.path.exists(config.FAISS_INDEX_PATH):
        store.load(config.FAISS_INDEX_PATH)
    return store

corpus = load_corpus()
base_store = load_vector_store()

st.title("🚨 When Retrieval Fails: Failure Exploration Console")
st.markdown("*An interactive diagnostic console exposing the internal retrieval vulnerabilities and generation boundaries of our evaluated RAG architecture.*")

tab1, tab2, tab3 = st.tabs(["⚡ Query Playground", "🔍 Failure Case Viewer", "📊 Metric Snapshot"])

with tab1:
    st.header("Query Playground")
    st.markdown("Test the RAG pipeline dynamically. Inject irrelevant Distractor Documents (Noise) or scale the Context Window ($K$).")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Experimental Controls")
        query = st.text_input("Enter Question:", "Who authored the laws of motion?")
        k_val = st.slider("Context Vectors (K)", min_value=1, max_value=10, value=3)
        noise_level = st.slider("Noise Level (Distractor Docs)", min_value=0, max_value=500, value=0)
        
        run_btn = st.button("Run Evaluation Pipeline", type="primary")
        
        st.caption("This triggers the actual Gemini 2.5 Flash and Embedding endpoints sequentially, mapping to the FAISS internal search.")
        
    with col2:
        if run_btn and query:
            if not corpus:
                st.error("Corpus not found. Ensure `data/load_squad.py` has run.")
            else:
                with st.spinner("Embedding query and executing Gemini FAISS retrieval..."):
                    query_emb = get_embedding(query)
                    
                    # Search base store
                    distances, indices = base_store.search(query_emb, k=k_val)
                    
                    retrieved_chunks = []
                    for i, idx in enumerate(indices[0]):
                        doc = next((c for c in corpus if str(c['id']) == str(base_store.index_map.get(str(idx)))), None)
                        if doc: retrieved_chunks.append({"chunk": doc['chunk'], "score": distances[0][i]})
                    
                    # Noise Injection Simulation: Pad context randomly with distractor corpus chunks
                    if noise_level > 0:
                        st.warning(f"⚠️ Injected {noise_level} distractor contexts into the prompt to stress test semantic focus.")
                        shuffled_corpus = random.sample(corpus, min(noise_level, len(corpus)))
                        for doc in shuffled_corpus:
                            retrieved_chunks.append({"chunk": doc['chunk'], "score": 0.0}) # Artificial noise
                    
                with st.spinner("Executing LLM generation and Faithfulness judge..."):
                    filtered_context = retrieved_chunks[: (k_val + noise_level)] 
                    final_answer = generate_answer(query, filtered_context)
                    
                    is_hall = is_hallucinated(final_answer)
                
                st.subheader("Model Output")
                st.info(final_answer)
                
                if is_hall:
                    st.error("🚨 HALLUCINATION DETECTED: The generation diverged from factual constraint bounds.")
                else:
                    st.success("✅ Output deemed FAITHFUL or successfully refused.")
                    
                st.subheader("Retrieval Transparency View")
                st.markdown(f"Displaying top-{k_val} retrieved chunks prior to scaling.")
                for i, chunk in enumerate(retrieved_chunks[:k_val]):
                    st.markdown(f"**Chunk {i+1}** (Cosine Similarity IP score: `{float(chunk['score']):.4f}`)")
                    st.caption(chunk["chunk"])

with tab2:
    st.header("Failure Case Viewer")
    st.markdown("Select a hand-curated qualitative failure generated during the Phase 2 adversarial injection test.")
    
    cases = {
        "Clean Hit (Phase 1 Baseline)": {
            "q": "In what country is Normandy located?", 
            "gt": "France", 
            "out": "Normandy is located in France.", 
            "eval": "✅ Exact Match",
            "desc": "Standard unperturbed RAG behavior executing correctly under K=3 parameters."
        },
        "Distractor Hallucination (+500 Noise Docs)": {
            "q": "What year did the Mongols invade Poland?", 
            "gt": "1259", 
            "out": "The Mongols invaded Hungary in 1241.", 
            "eval": "🚨 False Assertion (Hallucination)",
            "desc": "The model's attention was fundamentally diluted by the 500 injected adversarial documents, causing it to grab the closest semantic artifact (Hungary) instead of explicitly recognizing it didn't hold the Poland index."
        },
        "Refusal (Unanswerable Question)": {
            "q": "Who was the first person to walk on Mars?", 
            "gt": "", 
            "out": "I cannot answer based on the context.", 
            "eval": "✅ Safe Refusal",
            "desc": "Correctly identifying zero-shot boundaries and safely refusing generation logic when missing exact context vectors."
        },
    }
    selection = st.selectbox("Select Phenomenon", list(cases.keys()))
    
    sel_case = cases[selection]
    st.warning(sel_case["desc"])
    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"**Query:**\n\n> {sel_case['q']}")
        st.markdown(f"**Ground Truth Target:**\n\n> `{sel_case['gt'] if sel_case['gt'] else 'UNANSWERABLE (Refusal Array Expected)'}`")
    with colB:
        st.markdown(f"**Model Response:**\n\n> {sel_case['out']}")
        st.markdown(f"**Empirical Verdict:**\n\n> **{sel_case['eval']}**")

with tab3:
    st.header("Metric Snapshot")
    st.markdown("Visualized empirical decay of the RAG pipeline.")
    
    col3, col4 = st.columns(2)
    
    if os.path.exists("results/overall_metrics.png"):
        col3.image("results/overall_metrics.png", caption="Hallucination & F1 Trends under Noise Scaling")
    else:
        col3.info("`overall_metrics.png` is currently generating in the persistent background script... Once output naturally completes, refresh this console to visualize.")
        
    if os.path.exists("results/context_size_impact.png"):
        col4.image("results/context_size_impact.png", caption="Lost in the Middle Impact over scale K")
    else:
        col4.info("`context_size_impact.png` is parsing against the SQuAD limits... Refresh console to load.")
