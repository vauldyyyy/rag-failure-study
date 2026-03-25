import json
import urllib.request
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.chunker import chunk_text

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
TEMP_JSON = os.path.join(config.DATA_DIR, "squad_dev.json")

def load_and_process_squad():
    print("Downloading SQuAD v2 dev set...")
    if not os.path.exists(TEMP_JSON):
        urllib.request.urlretrieve(SQUAD_URL, TEMP_JSON)
        
    with open(TEMP_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    qa_pairs = []
    corpus = []
    
    doc_id = 0
    print("Processing corpus and extracting 600 QA pairs...")
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            
            chunks = chunk_text(context, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for c in chunks:
                corpus.append({
                    "doc_id": str(doc_id),
                    "text": c
                })
                
            for qa in paragraph['qas']:
                if qa['answers']: 
                    ans_text = qa['answers'][0]['text']
                    qa_pairs.append({
                        "question": qa['question'],
                        "answer": ans_text,
                        "doc_id": str(doc_id)
                    })
                    
            doc_id += 1
            if len(qa_pairs) >= 600:
                break
        if len(qa_pairs) >= 600:
            break
            
    qa_subset = qa_pairs[:600]
    
    with open(config.SQUAD_SUBSET_PATH, 'w', encoding='utf-8') as f:
        json.dump(qa_subset, f, ensure_ascii=False, indent=2)
        
    with open(config.CORPUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(qa_subset)} QA pairs and {len(corpus)} chunks.")

if __name__ == '__main__':
    load_and_process_squad()
