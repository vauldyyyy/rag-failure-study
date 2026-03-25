import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.vector_store import VectorStore
from utils.embedder import get_embeddings_batch, get_embedding
from utils.llm import generate_answer
from utils.metrics import exact_match_score, f1_score, precision_at_k, is_hallucinated

def run_baseline():
    print("--- Phase 1: Baseline RAG Evaluation ---")
    
    store = VectorStore(dimension=3072)
    
    if os.path.exists(config.FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        store.load(config.FAISS_INDEX_PATH)
    else:
        print("Building FAISS index (this takes a moment)...")
        with open(config.CORPUS_PATH, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
            
        texts = [c["text"] for c in corpus]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = get_embeddings_batch(texts)
        store.add_embeddings(embeddings, corpus)
        store.save()
        print("Index saved.")

    with open(config.SQUAD_SUBSET_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    results = []
    em_total, f1_total, p_at_k_total = 0, 0, 0
    hallucination_count = 0
    
    print("Evaluating 600 queries...")
    
    for i, qa in enumerate(qa_pairs):
        q = qa["question"]
        gold_ans = qa["answer"]
        gold_doc_id = qa["doc_id"]
        
        q_emb = get_embedding(q)
        retrieved = store.search(q_emb, top_k=config.TOP_K)
        retrieved_doc_ids = [c["doc_id"] for c in retrieved]
        
        pred = generate_answer(q, retrieved)
        
        em = exact_match_score(pred, gold_ans)
        f1 = f1_score(pred, gold_ans)
        p_k = precision_at_k(retrieved_doc_ids, gold_doc_id)
        
        hallucinated = 0
        if em == 0 and f1 < 0.2:
            if is_hallucinated(pred) != 0:
                hallucinated = 1
                hallucination_count += 1
                
        em_total += em
        f1_total += f1
        p_at_k_total += p_k
        
        results.append({
            "question": q,
            "gold_answer": gold_ans,
            "prediction": pred,
            "em": em,
            "f1": f1,
            "p_at_k": p_k,
            "hallucination_flag": hallucinated
        })
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/600...")

    n = len(qa_pairs)
    summary = {
        "EM": em_total / n,
        "F1": f1_total / n,
        "P@K": p_at_k_total / n,
        "Hallucination_Rate": hallucination_count / n
    }
    
    print("\nBaseline Results:")
    print(json.dumps(summary, indent=2))
    
    out_path = os.path.join(config.RESULTS_DIR, "phase1_baseline.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

if __name__ == '__main__':
    run_baseline()
