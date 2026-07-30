import json
import os
import sys

import faiss
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class VectorStore:
    """Exact inner-product FAISS index over L2-normalised vectors (= cosine).

    IndexFlatIP is exact, not approximate, on purpose: an ANN index would add a
    second, uncontrolled source of retrieval error on top of the one being
    studied, and 1.4k chunks is far too small for ANN to be worth it.
    """

    def __init__(self, dimension: int = 3072):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_map = {}

    def add_embeddings(self, embeddings: list, chunk_metadata: list):
        if not embeddings:
            return
        if len(embeddings) != len(chunk_metadata):
            raise ValueError(
                f"{len(embeddings)} embeddings vs {len(chunk_metadata)} metadata rows"
            )

        emb_arr = np.array(embeddings).astype("float32")
        if not np.isfinite(emb_arr).all():
            raise ValueError("Non-finite values in embeddings; refusing to index.")
        faiss.normalize_L2(emb_arr)

        # Positions are assigned by the index itself, so key off ntotal rather than
        # the map size — they only agree if nothing has ever been added out of band.
        start_id = self.index.ntotal
        for i, meta in enumerate(chunk_metadata):
            self.chunk_map[str(start_id + i)] = meta

        self.index.add(emb_arr)

    def search(self, query_embedding: list, top_k: int):
        q_arr = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(q_arr)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_arr, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.chunk_map[str(idx)]
            results.append({
                "score": float(score),
                "text": meta["text"],
                "doc_id": meta["doc_id"],
                "chunk_id": meta.get("chunk_id", str(idx)),
                "title": meta.get("title", ""),
            })
        return results

    def save(self, index_path: str = config.FAISS_INDEX_PATH, chunk_map_path: str = None):
        faiss.write_index(self.index, index_path)
        if chunk_map_path is None:
            chunk_map_path = index_path + "_map.json"
        with open(chunk_map_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_map, f, ensure_ascii=False)

    def load(self, index_path: str = config.FAISS_INDEX_PATH, chunk_map_path: str = None):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No index at {index_path}. Run the build step first."
            )
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d

        if chunk_map_path is None:
            chunk_map_path = index_path + "_map.json"
        if not os.path.exists(chunk_map_path):
            raise FileNotFoundError(f"Index present but chunk map missing at {chunk_map_path}")
        with open(chunk_map_path, "r", encoding="utf-8") as f:
            self.chunk_map = json.load(f)

        if len(self.chunk_map) != self.index.ntotal:
            raise ValueError(
                f"Index/map mismatch: {self.index.ntotal} vectors, {len(self.chunk_map)} entries"
            )
        return self


def build_or_load(corpus_path: str, index_path: str, rebuild: bool = False) -> VectorStore:
    """Load the index for a corpus, embedding and building it once if needed."""
    from utils.embedder import get_embeddings_batch, EMBED_DIM, STUB_DIM, flush

    store = VectorStore(dimension=STUB_DIM if config.DRY_RUN else EMBED_DIM)
    map_path = index_path + "_map.json"

    if not rebuild and os.path.exists(index_path) and os.path.exists(map_path):
        store.load(index_path)
        print(f"Loaded index: {store.index.ntotal} vectors from {os.path.basename(index_path)}")
        return store

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Embedding {len(corpus)} chunks...")
    embeddings = get_embeddings_batch([c["text"] for c in corpus])
    store.add_embeddings(embeddings, corpus)
    store.save(index_path)
    flush()
    print(f"Built index: {store.index.ntotal} vectors -> {index_path}")
    return store
