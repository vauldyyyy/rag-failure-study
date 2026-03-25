import google.generativeai as genai
import time
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

genai.configure(api_key=config.GEMINI_API_KEY)

def get_embedding(text: str, model: str = config.EMBED_MODEL, max_retries: int = 3):
    text = text.replace("\n", " ")
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            return result['embedding']
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to embed text: {e}")
                return [0.0] * 3072
            time.sleep(2 ** attempt)

def get_embeddings_batch(texts: List[str], model: str = config.EMBED_MODEL, batch_size: int = 100):
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ") for t in texts[i:i+batch_size]]
        for attempt in range(3):
            try:
                result = genai.embed_content(
                    model=model,
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embeddings = result['embedding']
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Batch failed: {e}")
                    all_embeddings.extend([[0.0] * 3072] * len(batch))
                time.sleep(2 ** attempt)
                
    return all_embeddings
