import faiss
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class VectorStore:
    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        # Inner product acts like cosine similarity if vectors are normalized
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_map = {}
        
    def add_embeddings(self, embeddings: list, chunk_metadata: list):
        if not embeddings: return
        
        emb_arr = np.array(embeddings).astype('float32')
        faiss.normalize_L2(emb_arr)
        
        start_id = len(self.chunk_map)
        for i, meta in enumerate(chunk_metadata):
            self.chunk_map[str(start_id + i)] = meta
            
        self.index.add(emb_arr)
        
    def search(self, query_embedding: list, top_k: int):
        q_arr = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(q_arr)
        
        distances, indices = self.index.search(q_arr, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "score": float(dist),
                    "chunk": self.chunk_map[str(idx)]["text"],
                    "doc_id": self.chunk_map[str(idx)]["doc_id"],
                    "chunk_id": str(idx)
                })
        return results
        
    def save(self, index_path: str = config.FAISS_INDEX_PATH, chunk_map_path: str = None):
        faiss.write_index(self.index, index_path)
        if chunk_map_path is None:
            chunk_map_path = index_path + "_map.json"
        with open(chunk_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_map, f, ensure_ascii=False)
            
    def load(self, index_path: str = config.FAISS_INDEX_PATH, chunk_map_path: str = None):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        if chunk_map_path is None:
            chunk_map_path = index_path + "_map.json"
        if os.path.exists(chunk_map_path):
            with open(chunk_map_path, 'r', encoding='utf-8') as f:
                self.chunk_map = json.load(f)
