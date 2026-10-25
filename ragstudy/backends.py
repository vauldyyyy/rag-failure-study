"""Model backends.

* GeminiBackend      — generation and embeddings through the google-genai SDK.
* OpenAICompatBackend — generation via any OpenAI-compatible API (Groq,
                        OpenRouter, Ollama, OpenAI) for cross-model comparisons.
* MockBackend        — deterministic offline stand-in used by the test suite and
                        for dry runs. Its numbers are NOT results.

Failures raise instead of silently returning zero vectors or placeholder text:
an API error must never be scored as a model answer.
"""
import os
import re
import threading
import time
import zlib

import numpy as np

import config


class RateLimiter:
    """Spaces calls evenly; safe to share between worker threads."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.next_slot = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


def _with_retries(fn, retryable, max_tries=8):
    for attempt in range(max_tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if attempt == max_tries - 1 or not retryable(e):
                raise
            time.sleep(min(60, 2 ** attempt * 2))


# ----------------------------------------------------------------------------- Gemini
class GeminiBackend:
    def __init__(self, gen_model=config.GEN_MODEL, embed_model=config.EMBED_MODEL,
                 embed_dim=config.EMBED_DIM, thinking_budget=config.THINKING_BUDGET,
                 rpm=config.REQUESTS_PER_MINUTE):
        from google import genai
        from google.genai import types
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in .env (see .env.example).")
        self.client = genai.Client(api_key=key)
        self.types = types
        self.gen_model, self.embed_model, self.embed_dim = gen_model, embed_model, embed_dim
        self.thinking_budget = thinking_budget
        self.limiter = RateLimiter(rpm)
        self.name = f"gemini:{gen_model}"
        self.embed_name = f"gemini:{embed_model}:{embed_dim}"

    @staticmethod
    def _retryable(e):
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        return code in (408, 429, 500, 502, 503, 504) or code is None

    def generate(self, system: str, prompt: str, model: str = None) -> dict:
        t = self.types
        cfg = dict(system_instruction=system, temperature=0.0,
                   max_output_tokens=config.MAX_OUTPUT_TOKENS)
        if self.thinking_budget >= 0:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=self.thinking_budget)

        def call():
            self.limiter.wait()
            return self.client.models.generate_content(
                model=model or self.gen_model, contents=prompt,
                config=t.GenerateContentConfig(**cfg))

        resp = _with_retries(call, self._retryable)
        finish = None
        if resp.candidates:
            finish = str(resp.candidates[0].finish_reason)
        return {"text": (resp.text or "").strip(), "finish_reason": finish}

    def embed(self, texts, task: str):
        out = []
        for i in range(0, len(texts), 100):
            batch = [s.replace("\n", " ") for s in texts[i:i + 100]]

            def call():
                self.limiter.wait()
                return self.client.models.embed_content(
                    model=self.embed_model, contents=batch,
                    config=self.types.EmbedContentConfig(
                        task_type=task, output_dimensionality=self.embed_dim))

            resp = _with_retries(call, self._retryable)
            if len(resp.embeddings) != len(batch):
                raise RuntimeError("Embedding count mismatch")
            out.extend(e.values for e in resp.embeddings)
        return np.asarray(out, dtype=np.float32)


# ----------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatBackend:
    """Generation only; pair it with --embed gemini or --embed mock."""

    def __init__(self, model=config.OPENAI_MODEL, base_url=config.OPENAI_BASE_URL,
                 rpm=config.REQUESTS_PER_MINUTE):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY"))
        self.gen_model = model
        self.limiter = RateLimiter(rpm)
        self.name = f"openai:{model}"

    def generate(self, system: str, prompt: str, model: str = None) -> dict:
        def call():
            self.limiter.wait()
            return self.client.chat.completions.create(
                model=model or self.gen_model, temperature=0,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}])

        def retryable(e):
            return getattr(e, "status_code", None) in (408, 429, 500, 502, 503, 504)

        resp = _with_retries(call, retryable)
        choice = resp.choices[0]
        return {"text": (choice.message.content or "").strip(),
                "finish_reason": choice.finish_reason}

    def embed(self, texts, task):
        raise NotImplementedError("Use --embed gemini or --embed mock with this backend.")


# ----------------------------------------------------------------------------- Mock
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to for from by with and or is are was were be been what which "
            "who whom whose when where why how did do does that this these those it its as".split())


def _content_words(s):
    return [w for w in _WORD.findall(s.lower()) if w not in _STOP]


class MockBackend:
    """Deterministic, offline. Exists only so the pipeline can be tested without keys."""

    name = "mock:extractive"
    embed_name = "mock:hash512"
    gen_model = "mock"

    def embed(self, texts, task):
        dim = 512
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = _content_words(t)
            for tok in words + [a + "_" + b for a, b in zip(words, words[1:])]:
                out[i, zlib.crc32(tok.encode()) % dim] += 1.0
        return out

    def generate(self, system: str, prompt: str, model: str = None) -> dict:
        if "SUPPORTED" in system:  # faithfulness judge
            ans = re.search(r"Answer:\s*(.*)\Z", prompt, re.S).group(1).strip().lower()
            ctx = prompt.lower()
            return {"text": "SUPPORTED" if ans and ans in ctx.split("answer:")[0] else "UNSUPPORTED",
                    "finish_reason": "STOP"}
        question = re.search(r"Question:\s*(.*?)\n", prompt).group(1)
        q = set(_content_words(question))
        ctx = prompt.split("Question:")[0]
        best, best_score = "", 0.0
        for sent in re.split(r"(?<=[.!?])\s+", ctx):
            words = _content_words(sent)
            if not words:
                continue
            score = len(q & set(words)) / max(1, len(q))
            if score > best_score:
                best, best_score = sent, score
        allow_abstain = "UNANSWERABLE" in system
        if allow_abstain and best_score < 0.5:
            return {"text": "UNANSWERABLE", "finish_reason": "STOP"}
        span = [w for w in best.split() if w.lower().strip(".,;:()") not in q][:4]
        return {"text": " ".join(span).strip(" .,;:") or "UNANSWERABLE", "finish_reason": "STOP"}


def make_backends(gen: str, embed: str):
    gen_b = {"gemini": GeminiBackend, "openai": OpenAICompatBackend, "mock": MockBackend}[gen]()
    if embed == gen and hasattr(gen_b, "embed_name"):
        emb_b = gen_b
    else:
        emb_b = {"gemini": GeminiBackend, "mock": MockBackend}[embed]()
    return gen_b, emb_b
