"""Scoring for the failure study.

The central measurement decision here: **hallucination is never scored from the
answer alone.** The judge always sees the retrieved context, because "is this
claim unsupported?" is undefined without the evidence the model was given. Where
the label can be derived without a judge at all — an unanswerable SQuAD question
that the model answered anyway — it is, because a rule is cheaper and more
reproducible than an LLM call.

Outcome taxonomy, applied per item:

    ABSTAIN              model declined to answer
    CORRECT              token F1 vs gold >= F1_CORRECT_THRESHOLD
    UNSUPPORTED          asserted content the retrieved context does not support
    INCORRECT_GROUNDED   wrong, but the claim does trace to the retrieved context
                         (a retrieval failure, not a generation failure)

Only UNSUPPORTED counts as hallucination. Separating it from INCORRECT_GROUNDED is
the point of the study: pooling them is what makes "RAG fails when retrieval is
bad" look like a finding.
"""

import os
import re
import string
import sys
import time
from collections import Counter

import google.generativeai as genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.llm import is_fatal_error, limiter as llm_limiter, note_call
from utils.ratelimit import (DailyQuotaExhausted, is_daily_quota_error,
                             is_quota_error, quota_sleep)

genai.configure(api_key=config.GEMINI_API_KEY)

ABSTAIN = "ABSTAIN"
CORRECT = "CORRECT"
UNSUPPORTED = "UNSUPPORTED"
INCORRECT_GROUNDED = "INCORRECT_GROUNDED"
INVALID = "INVALID"          # blocked / errored generation, excluded from all rates


# --------------------------------------------------------------------------- #
# Lexical scoring (official SQuAD v2 definitions)
# --------------------------------------------------------------------------- #

def normalize_answer(s):
    if not s:
        return ""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_single(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    # Multiset overlap, not set overlap: a set drops repeated tokens and inflates
    # precision on long generations, which is exactly the regime this study is in.
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truths):
    """Max EM over all annotator answers, per official SQuAD scoring."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    if not prediction or not ground_truths:
        return 0
    norm_pred = normalize_answer(prediction)
    return int(any(norm_pred == normalize_answer(g) for g in ground_truths))


def f1_score(prediction, ground_truths):
    """Max token F1 over all annotator answers."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    if not prediction or not ground_truths:
        return 0.0
    return max(_f1_single(prediction, g) for g in ground_truths)


def contains_gold(prediction, ground_truths):
    """Does the prediction contain a gold answer span?

    The standard correctness criterion in the RAG literature, and necessary here:
    a generator answering in prose is penalised by token-F1 for words the reference
    span simply does not contain. Guarded by a length cap so a rambling answer that
    happens to include the gold string among many other claims is not credited.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    if not prediction or not ground_truths:
        return 0
    norm_pred = normalize_answer(prediction)
    if len(norm_pred.split()) > 30:
        return 0
    return int(any(normalize_answer(g) and normalize_answer(g) in norm_pred
                   for g in ground_truths))


def precision_at_k(retrieved_doc_ids, true_doc_id):
    return int(true_doc_id in retrieved_doc_ids)


def recall_at_k(retrieved_doc_ids, gold_doc_ids):
    """Share of gold documents present in the retrieved set (multi-hop aware)."""
    gold = [g for g in (gold_doc_ids or []) if g]
    if not gold:
        return 0.0
    return sum(g in retrieved_doc_ids for g in gold) / len(gold)


# --------------------------------------------------------------------------- #
# Abstention detection
# --------------------------------------------------------------------------- #

_REFUSAL_PATTERNS = [
    r"i cannot answer",
    r"i can(?:'|no)t answer",
    r"cannot be answered",
    r"unable to answer",
    r"i do(?:n't| not) know",
    r"not contained in the (?:given |provided |above )?context",
    r"not (?:present|found|mentioned|stated|specified|provided|available) in the (?:given |provided |above )?context",
    r"context does(?:n't| not) (?:contain|provide|mention|specify|include)",
    r"no (?:answer|information) (?:is )?(?:available|provided|present)",
    r"insufficient (?:context|information)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_abstention(prediction: str):
    """True when the response is a refusal rather than an assertion.

    A refusal phrase alone is not enough — "the context does not say when, but it
    was 1241" is an assertion wearing a hedge, and scoring it as an abstention
    would hide precisely the silent-failure behaviour under study. So a response
    counts as abstention only if the refusal opens the response, or the whole
    response is short enough that there is no room for a trailing claim.
    """
    if not prediction or not prediction.strip():
        return False

    text = prediction.strip()
    match = _REFUSAL_RE.search(text)
    if not match:
        return False

    opens_with_refusal = match.start() <= 40
    is_short = len(text.split()) <= 25
    return bool(opens_with_refusal and is_short) or (opens_with_refusal and text.count(".") <= 1)


# --------------------------------------------------------------------------- #
# LLM-as-a-judge (context-grounded)
# --------------------------------------------------------------------------- #

_JUDGE_PROMPT = """You are grading whether an ANSWER is supported by the CONTEXT it was given.

You are NOT grading whether the answer is true in the real world, and you are NOT
grading whether it matches any reference answer. Judge only this: is every factual
claim in the ANSWER stated in or directly entailed by the CONTEXT?

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}

Reply with exactly one word:
SUPPORTED   - every claim in the answer is stated in or entailed by the context
UNSUPPORTED - the answer asserts at least one thing the context does not support
REFUSAL     - the answer declines to answer and asserts nothing"""

# "UNSUPPORTED" contains "SUPPORTED" as a substring, so a naive `if label in text`
# scan is order-dependent: reorder the labels and every hallucination verdict
# silently becomes SUPPORTED, inverting the study's primary outcome with no error.
# Word boundaries make the match order-independent — \bSUPPORTED\b cannot match
# inside UNSUPPORTED because the N-S junction is not a word boundary.
_JUDGE_LABEL_RE = re.compile(r"\b(UNSUPPORTED|SUPPORTED|REFUSAL)\b")


def parse_judge_label(text: str) -> str:
    """Map a judge response to a verdict. Returns JUDGE_ERROR if none is present."""
    if not text:
        return "JUDGE_ERROR"
    match = _JUDGE_LABEL_RE.search(text.upper())
    return match.group(1) if match else "JUDGE_ERROR"


_judge_model = None


def _get_judge():
    global _judge_model
    if _judge_model is None:
        _judge_model = genai.GenerativeModel(config.JUDGE_MODEL)
    return _judge_model


def judge_faithfulness(question: str, contexts: list, prediction: str, max_retries: int = 5):
    """Return SUPPORTED | UNSUPPORTED | REFUSAL | JUDGE_ERROR."""
    context_text = "\n\n".join(
        f"Context {i + 1}:\n{c['text']}" for i, c in enumerate(contexts)
    ) or "(no context was retrieved)"

    if config.DRY_RUN:
        from utils.stub_llm import stub_judge
        return stub_judge(question, contexts, prediction)

    prompt = _JUDGE_PROMPT.format(
        context=context_text, question=question, answer=prediction
    )

    from utils.ratelimit import TRANSIENT_RETRIES, is_transient_server_error, transient_sleep
    transient = 0
    attempt = -1
    while attempt < max_retries - 1:
        attempt += 1
        try:
            llm_limiter.acquire(1)
            note_call()
            response = _get_judge().generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0),
            )
            return parse_judge_label(response.text)
        except DailyQuotaExhausted:
            raise
        except Exception as e:
            if is_daily_quota_error(e):
                raise DailyQuotaExhausted(f"Daily quota exhausted during judging: {e}") from e
            if is_transient_server_error(e) and transient < TRANSIENT_RETRIES:
                transient += 1
                attempt -= 1
                time.sleep(transient_sleep(transient))
                continue
            if is_quota_error(e):
                llm_limiter.note_rejection()
                if attempt == max_retries - 1:
                    return "JUDGE_ERROR"
                time.sleep(quota_sleep(attempt))
                continue
            if is_fatal_error(e):
                raise RuntimeError(f"Unrecoverable judge error (not retrying): {e}") from e
            if attempt == max_retries - 1:
                print(f"Judge error: {e}")
                return "JUDGE_ERROR"
            time.sleep(min(60, (2 ** attempt) * 4))
    return "JUDGE_ERROR"


# --------------------------------------------------------------------------- #
# Item-level outcome
# --------------------------------------------------------------------------- #

def score_item(qa, prediction, status, contexts, use_judge=True):
    """Score one generation into the outcome taxonomy.

    `contexts` must be exactly the context list handed to the generator, after any
    noise perturbation — judging against the pre-perturbation retrieval would score
    a different experiment than the one that was run.
    """
    golds = qa.get("answers") or []
    answerable = qa.get("is_answerable", True)

    row = {
        "qid": qa.get("qid"),
        "question": qa["question"],
        "gold_answers": golds,
        "is_answerable": answerable,
        "prediction": prediction,
        "gen_status": status,
    }

    if status != "ok":
        row.update({"outcome": INVALID, "em": None, "f1": None, "abstained": None})
        return row

    em = exact_match_score(prediction, golds) if golds else 0
    f1 = f1_score(prediction, golds) if golds else 0.0
    has_gold = contains_gold(prediction, golds) if golds else 0
    abstained = is_abstention(prediction)

    gold_doc_ids = qa.get("gold_doc_ids") or [qa.get("gold_doc_id")]
    ctx_doc_ids = [c["doc_id"] for c in contexts]
    gold_in_context = any(g in ctx_doc_ids for g in gold_doc_ids if g)

    row.update({
        "em": em,
        "f1": round(f1, 4),
        "contains_gold": has_gold,
        "abstained": abstained,
        "gold_in_context": gold_in_context,
        # Cheap provenance on every row: which chunks reached the generator.
        # Enough to reconstruct an unperturbed context from the corpus.
        "context_chunk_ids": [c.get("chunk_id") for c in contexts],
    })

    if abstained:
        row["outcome"] = ABSTAIN
        row["judge"] = None
        return row

    if not answerable:
        # SQuAD v2 guarantees the gold paragraph contains no answer. The model
        # asserted something anyway, so no judge call is needed or wanted.
        row["outcome"] = UNSUPPORTED
        row["judge"] = "rule:unanswerable_asserted"
        return row

    # Correct if the answer either matches the reference span well (F1) or contains
    # it outright. See PREREGISTRATION.md, deviation 2026-07-31b — the F1-only rule
    # was preregistered and shown by the pilot to misclassify correct prose answers.
    if f1 >= config.F1_CORRECT_THRESHOLD or has_gold:
        row["outcome"] = CORRECT
        row["judge"] = None
        # Correct without the gold document in context = the answer came from the
        # model's weights, not from retrieval. Tracked because it inflates every
        # accuracy number in a way retrieval ablations are supposed to remove.
        row["parametric"] = not gold_in_context
        return row

    if use_judge:
        verdict = judge_faithfulness(row["question"], contexts, prediction)
        row["judge"] = verdict
        # Store the exact evidence the judge ruled on. Without it a verdict cannot
        # be re-audited, re-scored by a second judge, or checked against a human
        # label — and the headline metric of this study is judge-derived. Stored
        # only on judged rows, which are a minority, so results files stay small.
        row["judged_context"] = [
            {"doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id"),
             "perturbed": c.get("perturbed"), "text": c.get("text", "")}
            for c in contexts
        ]
        if verdict == "UNSUPPORTED":
            row["outcome"] = UNSUPPORTED
        elif verdict == "REFUSAL":
            row["outcome"] = ABSTAIN
        elif verdict == "SUPPORTED":
            row["outcome"] = INCORRECT_GROUNDED
        else:
            row["outcome"] = INVALID
    else:
        row["judge"] = None
        row["outcome"] = UNSUPPORTED if not gold_in_context else INCORRECT_GROUNDED

    return row


def aggregate(rows):
    """Condition-level rates. Rates are computed over valid generations only."""
    valid = [r for r in rows if r["outcome"] != INVALID]
    n = len(valid)
    if n == 0:
        return {"n": 0, "n_invalid": len(rows)}

    ans = [r for r in valid if r["is_answerable"]]
    unans = [r for r in valid if not r["is_answerable"]]

    def rate(subset, pred):
        return round(sum(1 for r in subset if pred(r)) / len(subset), 4) if subset else None

    summary = {
        "n": n,
        "n_invalid": len(rows) - n,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "EM": round(sum(r["em"] for r in ans) / len(ans), 4) if ans else None,
        "F1": round(sum(r["f1"] for r in ans) / len(ans), 4) if ans else None,
        "contains_gold_rate": rate(ans, lambda r: r.get("contains_gold")),
        "abstention_rate": rate(valid, lambda r: r["outcome"] == ABSTAIN),
        "abstention_rate_answerable": rate(ans, lambda r: r["outcome"] == ABSTAIN),
        "hallucination_rate": rate(valid, lambda r: r["outcome"] == UNSUPPORTED),
        "hallucination_rate_answerable": rate(ans, lambda r: r["outcome"] == UNSUPPORTED),
        "hallucination_rate_unanswerable": rate(unans, lambda r: r["outcome"] == UNSUPPORTED),
        "incorrect_grounded_rate": rate(valid, lambda r: r["outcome"] == INCORRECT_GROUNDED),
        "gold_in_context_rate": rate(valid, lambda r: r.get("gold_in_context")),
        "parametric_correct_rate": rate(ans, lambda r: r.get("parametric") is True),
        "judge_self_eval": config.JUDGE_SELF_EVAL,
    }

    # The headline behaviour: among items the model got wrong, how often did it
    # assert rather than abstain? This is the "silent failure" rate.
    wrong = [r for r in valid if r["outcome"] in (UNSUPPORTED, INCORRECT_GROUNDED, ABSTAIN)]
    summary["silent_failure_rate"] = rate(wrong, lambda r: r["outcome"] != ABSTAIN)
    return summary
