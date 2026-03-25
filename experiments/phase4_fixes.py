import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.vector_store import VectorStore
from utils.embedder import get_embedding
from utils.llm import generate_answer
from utils.metrics import exact_match_score, f1_score

def run_fixes_exp():
    print("--- Phase 4: Strict RAG Configuration Evaluation ---")
    store = VectorStore(dimension=3072)
    store.load(config.FAISS_INDEX_PATH)
    
    with open(config.SQUAD_SUBSET_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    em_total, f1_total = 0, 0
    
    strict_prompt = (
        "You are a strict technical assistant. You must ONLY answer using the exact facts provided in the context below. "
        "If the context does not explicitly contain the answer, you must output exactly 'I do not know'. "
        "Do not hallucinate, deduce, or infer outside the given text."
    )
    
    for i, qa in enumerate(qa_pairs):
        q, gold_ans = qa["question"], qa["answer"]
        q_emb = get_embedding(q)
        retrieved = store.search(q_emb, top_k=config.TOP_K)
        
        pred = generate_answer(q, retrieved, system_prompt=strict_prompt)
        em, f1 = exact_match_score(pred, gold_ans), f1_score(pred, gold_ans)
        
        em_total += em
        f1_total += f1
        if (i+1) % 100 == 0: print(f" Processed Phase 4 {i+1}/600...")

    n = len(qa_pairs)
    summary = {"EM": em_total / n, "F1": f1_total / n}
    print("\nPhase 4 Results:", summary)
    
    with open(os.path.join(config.RESULTS_DIR, "phase4_fixes.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    run_fixes_exp()
