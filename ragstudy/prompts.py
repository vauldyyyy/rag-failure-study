"""Prompt templates. Three answering conditions differ only in their abstention instruction.

open    : no permission to abstain — measures the model's natural tendency to answer.
default : may abstain with the token UNANSWERABLE.
strict  : must abstain unless the answer is explicitly stated; no inference or prior knowledge.
"""

ABSTAIN_TOKEN = "UNANSWERABLE"

_SHORT = "Reply with the shortest possible answer (a few words), with no explanation."

SYSTEM = {
    "open": f"Answer the question using the provided context. {_SHORT}",
    "default": (f"Answer the question using the provided context. {_SHORT} "
                f"If the context does not contain the answer, reply exactly: {ABSTAIN_TOKEN}"),
    "strict": (f"Answer ONLY with facts stated explicitly in the provided context. Do not infer, "
               f"guess or use prior knowledge. {_SHORT} If the context does not explicitly state "
               f"the answer, reply exactly: {ABSTAIN_TOKEN}"),
}


def format_context(passages) -> str:
    return "\n\n".join(f"[{i}] {p['title'].replace('_', ' ')}\n{p['text']}"
                       for i, p in enumerate(passages, 1))


def answer_prompt(question: str, passages) -> str:
    return f"Context:\n{format_context(passages)}\n\nQuestion: {question}\nAnswer:"


JUDGE_SYSTEM = ("You are a strict fact-checker. Decide whether every factual claim in the Answer is "
                "directly supported by the Context. Reply with exactly one word: SUPPORTED or UNSUPPORTED.")


def judge_prompt(question: str, passages, answer: str) -> str:
    return (f"Context:\n{format_context(passages)}\n\nQuestion: {question}\n"
            f"Answer: {answer}")
