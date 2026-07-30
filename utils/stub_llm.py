"""Deterministic stub generator for `--dry-run`.

Purpose: execute every branch of the pipeline — all phases, all outcome labels,
the statistics, the figures, the LaTeX tables — without an API key and without
spending quota, so that plumbing bugs surface before the real run rather than
four hours into it.

**Numbers produced in dry-run mode are meaningless.** The stub's behaviour is a
crude, hand-written function of "is the gold document in the context", chosen to
exercise code paths, not to model an LLM. Every results file written in this mode
carries `"synthetic": true`, `verify_paper.py` refuses to build a submission from
such a file, and figures generated from them are stamped SYNTHETIC.

It is deliberately *not* tuned to produce the study's hypotheses. If a dry-run
figure looks like it supports H1, that is an artefact of this file, not evidence.
"""

import hashlib
import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

ABSTAIN_TEXT = "I cannot answer based on the context."


def _unit(*parts) -> float:
    """Deterministic pseudo-random float in [0,1) from the given key parts."""
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(1 << 48)


def stub_generate(query, contexts, system_prompt=None, gold_answers=None,
                  gold_doc_ids=None, is_answerable=True):
    """Return {"text", "status"} shaped exactly like utils.llm.generate_answer."""
    ctx_ids = [c.get("doc_id") for c in contexts]
    gold_ids = [g for g in (gold_doc_ids or []) if g]
    gold_present = any(g in ctx_ids for g in gold_ids)

    roll = _unit(query, len(contexts), system_prompt or "", gold_present, is_answerable)
    # A stricter system prompt shifts the stub toward abstention, so Phase 4 has
    # something to measure. The size of the shift is arbitrary.
    abstain_bias = 0.25 if system_prompt and "cannot answer" in system_prompt.lower() else 0.0

    if not is_answerable:
        # Unanswerable: abstain sometimes, otherwise assert something from context.
        return {"text": ABSTAIN_TEXT if roll < 0.45 + abstain_bias
                else _fabricate(query, contexts, roll), "status": "ok"}

    if gold_present and gold_answers:
        if roll < 0.72:
            return {"text": str(gold_answers[0]), "status": "ok"}
        if roll < 0.80 + abstain_bias:
            return {"text": ABSTAIN_TEXT, "status": "ok"}
        return {"text": _fabricate(query, contexts, roll), "status": "ok"}

    # Gold absent on an answerable question: abstain, guess from context, or --
    # occasionally -- get it right anyway, which is the parametric-answer path the
    # scorer needs to see exercised.
    if roll < 0.35 + abstain_bias:
        return {"text": ABSTAIN_TEXT, "status": "ok"}
    if roll < 0.42 and gold_answers:
        return {"text": str(gold_answers[0]), "status": "ok"}
    return {"text": _fabricate(query, contexts, roll), "status": "ok"}


def _fabricate(query, contexts, roll):
    """Assemble a plausible-looking wrong answer out of context tokens."""
    words = []
    for c in contexts:
        words.extend(_WORD_RE.findall(c.get("text", ""))[:40])
    if not words:
        return "The answer is not determinable."
    start = int(roll * max(len(words) - 6, 1))
    return " ".join(words[start:start + 6]).strip() or "Unknown."


def stub_judge(question, contexts, prediction):
    """Context-grounded judge stub: SUPPORTED if the answer's content words appear."""
    if prediction.strip().lower().startswith("i cannot answer"):
        return "REFUSAL"
    haystack = " ".join(c.get("text", "") for c in contexts).lower()
    tokens = [w.lower() for w in _WORD_RE.findall(prediction) if len(w) > 3]
    if not tokens:
        return "UNSUPPORTED"
    hit_rate = sum(t in haystack for t in tokens) / len(tokens)
    return "SUPPORTED" if hit_rate >= 0.6 else "UNSUPPORTED"
