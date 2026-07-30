import hashlib
import json
import os
import sys
import time
from typing import List

import google.generativeai as genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.llm import is_fatal_error
from utils.ratelimit import (DailyQuotaExhausted, RateLimiter, is_daily_quota_error,
                             is_quota_error, quota_sleep)

genai.configure(api_key=config.GEMINI_API_KEY)

_limiter = RateLimiter(config.EMBED_RPM, name="embed")

EMBED_DIM = 3072
STUB_DIM = 512  # dry-run only

# --------------------------------------------------------------------------- #
# Local backend (sentence-transformers)
# --------------------------------------------------------------------------- #
_st_model = None


def _get_local_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading local embedder {config.LOCAL_EMBED_MODEL} (CPU)...")
        _st_model = SentenceTransformer(config.LOCAL_EMBED_MODEL, device="cpu")
    return _st_model


def embed_dim():
    """Index dimension for the active backend."""
    if config.DRY_RUN:
        return STUB_DIM
    if config.EMBED_BACKEND == "local":
        return config.LOCAL_EMBED_DIM
    return EMBED_DIM


def uses_api() -> bool:
    return not config.DRY_RUN and config.EMBED_BACKEND != "local"


def _stub_embedding(text: str):
    """Hashed bag-of-words vector — deterministic, offline, no API.

    Used only under --dry-run. Retrieval quality is poor compared to a real
    embedding model, which is fine and even useful: it guarantees the pipeline is
    exercised on items where the gold document was *not* retrieved.
    """
    import re
    vec = [0.0] * STUB_DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) < 3:
            continue
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % STUB_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec

# Query embeddings are re-used across every experimental condition, so they are
# cached on disk. This is not an optimisation detail: it also guarantees that two
# conditions compared against each other saw byte-identical query vectors.
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        if os.path.exists(config.QUERY_EMB_CACHE):
            with open(config.QUERY_EMB_CACHE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _flush_cache():
    if _cache is not None:
        tmp = config.QUERY_EMB_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f)
        os.replace(tmp, config.QUERY_EMB_CACHE)


def _key(text: str, task_type: str) -> str:
    return hashlib.sha1(f"{config.EMBED_MODEL}|{task_type}|{text}".encode("utf-8")).hexdigest()


def get_embedding(text: str, model: str = config.EMBED_MODEL,
                  task_type: str = "RETRIEVAL_QUERY", max_retries: int = 5,
                  use_cache: bool = True):
    text = text.replace("\n", " ")
    if config.DRY_RUN:
        return _stub_embedding(text)
    if config.EMBED_BACKEND == "local":
        return _get_local_model().encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
    cache = _load_cache() if use_cache else {}
    k = _key(text, task_type)
    if k in cache:
        return cache[k]

    for attempt in range(max_retries):
        try:
            _limiter.acquire(1)
            result = genai.embed_content(model=model, content=text, task_type=task_type)
            emb = result["embedding"]
            if use_cache:
                cache[k] = emb
                if len(cache) % 50 == 0:
                    _flush_cache()
            return emb
        except Exception as e:
            # Returning a zero vector on failure would silently corrupt every
            # downstream similarity score, so this always raises rather than degrades.
            if is_daily_quota_error(e):
                _flush_cache()
                raise DailyQuotaExhausted(f"Daily embedding quota exhausted: {e}") from e
            if is_quota_error(e):
                _limiter.note_rejection()
                if attempt == max_retries - 1:
                    _flush_cache()
                    raise RuntimeError(f"Embedding quota still exhausted after "
                                       f"{max_retries} attempts: {e}") from e
                time.sleep(quota_sleep(attempt))
                continue
            if is_fatal_error(e):
                raise RuntimeError(f"Unrecoverable embedding error (not retrying): {e}") from e
            if attempt == max_retries - 1:
                raise RuntimeError(f"Embedding failed after {max_retries} attempts: {e}") from e
            time.sleep(min(60, (2 ** attempt) * 2))


def get_embeddings_batch(texts: List[str], model: str = config.EMBED_MODEL,
                         batch_size: int = 100, task_type: str = "RETRIEVAL_DOCUMENT"):
    if config.DRY_RUN:
        return [_stub_embedding(t) for t in texts]

    if config.EMBED_BACKEND == "local":
        model = _get_local_model()
        return model.encode(
            [t.replace("\n", " ") for t in texts],
            batch_size=64, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).tolist()

    # --- API backend: cache every vector so a quota stop is resumable ---------
    # The first real run died 760 of 1330 chunks into the index build and lost all
    # of it, which meant the next day would re-spend the same quota and stop in the
    # same place — the index could never be completed. Document vectors are now
    # cached on disk exactly like query vectors.
    doc_cache = _load_cache()
    keys = [_key(t.replace("\n", " "), task_type) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in doc_cache]
    if len(missing_idx) < len(texts):
        print(f"  {len(texts) - len(missing_idx)}/{len(texts)} chunks already cached")
    if not missing_idx:
        return [doc_cache[k] for k in keys]

    keys_missing = [keys[i] for i in missing_idx]
    texts = [texts[i] for i in missing_idx]
    all_embeddings = []
    # A batched call is billed per item, so keep batches at or under the per-minute
    # allowance; a larger batch cannot succeed even once.
    batch_size = min(batch_size, config.EMBED_RPM)

    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ") for t in texts[i:i + batch_size]]
        for attempt in range(6):
            try:
                _limiter.acquire(len(batch))
                result = genai.embed_content(model=model, content=batch, task_type=task_type)
                embeddings = result["embedding"]
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"API returned {len(embeddings)} embeddings for {len(batch)} inputs"
                    )
                all_embeddings.extend(embeddings)
                # Persist as we go, so a quota stop keeps everything bought so far.
                for k, emb in zip(keys_missing[len(all_embeddings) - len(embeddings):
                                              len(all_embeddings)], embeddings):
                    doc_cache[k] = emb
                _flush_cache()
                break
            except Exception as e:
                if is_daily_quota_error(e):
                    _flush_cache()
                    raise DailyQuotaExhausted(
                        f"Daily embedding quota exhausted after {len(all_embeddings)} "
                        f"of {len(texts)} new chunks ({len(doc_cache)} cached total). "
                        f"Progress is saved — re-running tomorrow resumes here. "
                        f"Or switch to the local backend: EMBED_BACKEND=local in .env, "
                        f"which needs no quota at all."
                    ) from e
                if is_quota_error(e):
                    _limiter.note_rejection()
                    if attempt == 5:
                        raise RuntimeError(
                            f"Embedding batch {i}-{i + len(batch)} still quota-blocked "
                            f"after 6 attempts: {e}"
                        ) from e
                    time.sleep(quota_sleep(attempt))
                    continue
                if is_fatal_error(e):
                    raise RuntimeError(f"Unrecoverable embedding error (not retrying): {e}") from e
                if attempt == 5:
                    raise RuntimeError(
                        f"Embedding batch {i}-{i + len(batch)} failed after 6 attempts: {e}"
                    ) from e
                time.sleep(min(60, (2 ** attempt) * 2))

        done = min(i + batch_size, len(texts))
        eta = (len(texts) - done) / max(config.EMBED_RPM, 1)
        print(f"  embedded {done}/{len(texts)}  (~{eta:.0f} min left at "
              f"{config.EMBED_RPM}/min)   ", end="\r")

    print()
    _flush_cache()
    # Return in the caller's original order, mixing cached and freshly fetched.
    return [doc_cache[k] for k in keys]


def flush():
    _flush_cache()
