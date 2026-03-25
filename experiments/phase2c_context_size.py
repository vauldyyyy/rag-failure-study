import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.vector_store import VectorStore
from utils.embedder import get_embedding
from utils.llm import generate_answer
from utils.metrics import exact_match_score, f1_score

def run_context_size_exp():
    print("--- Phase 2c: Context Size RAG Evaluation ---")
    store = VectorStore(dimension=3072)
    store.load(config.FAISS_INDEX_PATH)
    
    with open(config.SQUAD_SUBSET_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    k_vals = [1, 5]
    all_summaries = {}
    
    for k in k_vals:
        print(f"\nEvaluating Context Size Top-K = {k}")
        em_total, f1_total = 0, 0
        details = []
        for i, qa in enumerate(qa_pairs):
            q, gold_ans = qa["question"], qa["answer"]
            q_emb = get_embedding(q)
            retrieved = store.search(q_emb, top_k=k)
            
            pred = generate_answer(q, retrieved)
            em, f1 = exact_match_score(pred, gold_ans), f1_score(pred, gold_ans)
            
            em_total += em
            f1_total += f1
            details.append({"q": q, "em": em, "f1": f1})
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/600...")
            
        n = len(qa_pairs)
        summary = {"EM": em_total / n, "F1": f1_total / n}
        all_summaries[f"top_{k}"] = summary
        print(f"Top-{k} Results:", summary)
        
    with open(os.path.join(config.RESULTS_DIR, "phase2c_context_size.json"), 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2)

if __name__ == '__main__':
    run_context_size_exp()
