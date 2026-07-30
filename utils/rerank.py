"""Mitigations that act on the context before generation.

The lexical reranker is deliberately dependency-free and API-free. A cross-encoder
(ms-marco-MiniLM) would very likely rerank better, but it would also add a second
neural model to the pipeline and make the mitigation result a statement about that
checkpoint. A BM25-style scorer isolates the question actually being asked: does
*any* relevance filter between retrieval and generation recover the loss, or does
the generator fail regardless of context ordering?
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "whom", "when", "where", "why", "how",
    "did", "do", "does", "it", "its", "this", "that", "by", "as", "at", "from",
    "with", "be", "been", "has", "have", "had",
}


def _tokens(text):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def bm25_rerank(query, contexts, k1=1.5, b=0.75, keep=None):
    """Reorder `contexts` by BM25 against the query; optionally keep the top `keep`.

    IDF is computed over the candidate set itself rather than the full corpus. With
    a handful of candidates this is a within-context relevance signal, not a global
    one — which is the correct scope, since the reranker only ever sees the k chunks
    retrieval already returned.
    """
    if not contexts:
        return contexts

    docs = [_tokens(c["text"]) for c in contexts]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n if n else 0.0
    df = Counter()
    for d in docs:
        df.update(set(d))

    q_terms = _tokens(query)
    scored = []
    for ctx, doc in zip(contexts, docs):
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * (dl / avgdl if avgdl else 1))
            score += idf * (tf[term] * (k1 + 1)) / denom
        scored.append((score, ctx))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [dict(ctx, rerank_score=round(s, 4)) for s, ctx in scored]
    return out[:keep] if keep else out


def make_reranker(keep=None):
    def _rerank(query, contexts):
        return bm25_rerank(query, contexts, keep=keep)
    return _rerank
