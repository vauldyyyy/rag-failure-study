import os
import sys
import time

import google.generativeai as genai
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.ratelimit import (DailyQuotaExhausted, RateLimiter, TRANSIENT_RETRIES,
                             is_daily_quota_error, is_quota_error,
                             is_transient_server_error, quota_sleep, transient_sleep)

genai.configure(api_key=config.GEMINI_API_KEY)

# Shared by generation and the judge — they draw on the same per-model quota.
limiter = RateLimiter(config.LLM_RPM, name="llm")
_calls_today = {"n": 0}


def note_call():
    _calls_today["n"] += 1
    if config.LLM_RPD and _calls_today["n"] >= config.LLM_RPD:
        raise DailyQuotaExhausted(
            f"Client-side daily cap reached ({config.LLM_RPD} calls). "
            f"Checkpoints are saved; resume tomorrow."
        )


def calls_made():
    return _calls_today["n"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the following context to answer the question. "
    "Answer with the shortest span that answers the question — a few words, not a "
    "sentence. Do not restate the question or add explanation. "
    "If the answer is not contained in the context, say 'I cannot answer based on the context'."
)
# The brevity instruction is a measurement requirement, not a style preference.
# SQuAD gold answers are bare spans, so token-F1 against a full sentence scores a
# correct answer near 0.3 on precision alone. Observed in the pilot: gold "The
# mermaid", model "The mermaid (syrenka) is Warsaw's symbol.", F1 0.33.

# Sent to the model verbatim when a condition requires the model to abstain rather
# than guess. Kept as a constant so the abstention detector in metrics.py and the
# prompt can never drift apart.
ABSTAIN_STRING = "I cannot answer based on the context"

_MODELS = {}

# Errors that will never resolve by waiting. Retrying these just turns a bad key
# or a wrong model name into a five-minute silent hang.
_FATAL_MARKERS = (
    "api key", "api_key", "unauthenticated", "permission_denied", "permission denied",
    "invalid_argument", "is not found", "not found for api version", "unsupported",
)


def is_fatal_error(e) -> bool:
    msg = str(e).lower()
    return any(m in msg for m in _FATAL_MARKERS)


def supports_system_instruction(model_name: str) -> bool:
    """Gemma checkpoints served through the Gemini API reject system_instruction
    ("Developer instruction is not enabled"). For them the identical instruction
    text is prepended to the user turn instead; see generate_answer."""
    return "gemma" not in model_name.lower()


_REST_URL = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"


def _rest_generate(model_name: str, user_prompt: str, temperature: float):
    """Direct REST call, used for Gemma.

    Gemma 4 returns its reasoning as separate parts marked "thought": true. The
    pinned google-generativeai SDK drops that flag, so response.text glues the
    reasoning onto the answer and every Gemma answer would be scored on a page of
    chain-of-thought instead of a span. Here thought parts are discarded and only
    the answer parts are kept. Errors are raised with the HTTP status and Google's
    status string in the message so the existing quota/fatal classifiers apply.
    """
    name = model_name if model_name.startswith("models/") else f"models/{model_name}"
    r = requests.post(
        _REST_URL.format(model=name),
        headers={"x-goog-api-key": config.GEMINI_API_KEY,
                 "Content-Type": "application/json"},
        json={"contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
              "generationConfig": {"temperature": temperature}},
        timeout=60,
    )
    if r.status_code != 200:
        try:
            err = r.json().get("error", {})
        except ValueError:
            err = {}
        raise RuntimeError(f"{r.status_code} {err.get('status', '')} "
                           f"{err.get('message', r.text[:200])}")
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        return {"text": "", "status": "blocked", "detail": str(reason)}
    parts = (cands[0].get("content") or {}).get("parts") or []
    answer = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not answer.strip():
        return {"text": "", "status": "blocked",
                "detail": str(cands[0].get("finishReason"))}
    return {"text": answer, "status": "ok"}


def _get_model(model_name: str, system_prompt: str):
    native = supports_system_instruction(model_name)
    key = (model_name, system_prompt if native else None)
    if key not in _MODELS:
        _MODELS[key] = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt if native else None,
        )
    return _MODELS[key]


def build_prompt(query: str, retrieved_contexts: list) -> str:
    context_text = "\n\n".join(
        f"Context {i + 1}:\n{c['text']}" for i, c in enumerate(retrieved_contexts)
    )
    return (
        "Context information is below.\n"
        "---------------------\n"
        f"{context_text}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        f"Query: {query}\n"
        "Answer: "
    )


def generate_answer(query: str, retrieved_contexts: list,
                    model_name: str = config.LLM_MODEL,
                    system_prompt: str = None,
                    temperature: float = 0.0,
                    max_retries: int = 5):
    """Return {"text", "status"}.

    status is one of ok | blocked | error. It is carried into the results files
    because a safety-blocked or errored generation is not evidence of anything and
    must be excluded from rates rather than silently scored as a wrong answer.
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    model = _get_model(model_name, system_prompt)
    user_prompt = build_prompt(query, retrieved_contexts)
    if not supports_system_instruction(model_name):
        user_prompt = f"{system_prompt}\n\n{user_prompt}"

    transient = 0
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            limiter.acquire(1)
            note_call()
            if not supports_system_instruction(model_name):
                return _rest_generate(model_name, user_prompt, temperature)
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            try:
                return {"text": response.text, "status": "ok"}
            except (ValueError, AttributeError):
                # No text part: safety block, recitation stop, or empty candidate.
                reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
                return {"text": "", "status": "blocked", "detail": str(reason)}
        except DailyQuotaExhausted:
            raise
        except Exception as e:
            # A per-day cap will not clear by waiting, so stop the run cleanly and
            # let the caller checkpoint rather than burn attempts.
            if is_daily_quota_error(e):
                raise DailyQuotaExhausted(f"Daily generation quota exhausted: {e}") from e
            if is_transient_server_error(e) and transient < TRANSIENT_RETRIES:
                # Server hiccup: does not consume one of the ordinary attempts.
                transient += 1
                attempt -= 1
                print(f"    [gen] transient {str(e)[:40]!r}, retry {transient}/{TRANSIENT_RETRIES}",
                      file=sys.stderr, flush=True)
                time.sleep(transient_sleep(transient))
                continue
            if is_quota_error(e):
                limiter.note_rejection()
                if attempt == max_retries:
                    return {"text": "", "status": "error", "detail": f"quota: {e}"}
                print(f"    [gen] quota {str(e)[:40]!r}", file=sys.stderr, flush=True)
                time.sleep(quota_sleep(attempt - 1))
                continue
            if is_fatal_error(e):
                raise RuntimeError(f"Unrecoverable Gemini error (not retrying): {e}") from e
            if attempt == max_retries:
                print(f"LLM Error: {e}")
                return {"text": "", "status": "error", "detail": str(e)}
            time.sleep(min(60, (2 ** (attempt - 1)) * 4))
    return {"text": "", "status": "error", "detail": "retries exhausted"}
