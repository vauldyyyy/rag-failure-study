import os
import sys
import time

import google.generativeai as genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

genai.configure(api_key=config.GEMINI_API_KEY)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the following context to answer the question. "
    "If the answer is not contained in the context, say 'I cannot answer based on the context'."
)

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


def _get_model(model_name: str, system_prompt: str):
    key = (model_name, system_prompt)
    if key not in _MODELS:
        _MODELS[key] = genai.GenerativeModel(
            model_name=model_name, system_instruction=system_prompt
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

    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            if is_fatal_error(e):
                raise RuntimeError(f"Unrecoverable Gemini error (not retrying): {e}") from e
            if attempt == max_retries - 1:
                print(f"LLM Error: {e}")
                return {"text": "", "status": "error", "detail": str(e)}
            # Google AI free tier is 15 RPM; back off exponentially, capped at 60s.
            time.sleep(min(60, (2 ** attempt) * 4))
