import tiktoken

_ENCODINGS = {}


def _get_encoding(model: str):
    if model not in _ENCODINGS:
        try:
            _ENCODINGS[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODINGS[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODINGS[model]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, model: str = "gpt-3.5-turbo"):
    """Token-aware fixed-size splitter.

    tiktoken is used only as a stable, model-independent tokenizer for chunk
    boundaries; it is not the Gemini tokenizer, so chunk sizes are approximate
    in Gemini tokens. That is fine here because chunk size is held constant
    across every experimental condition.
    """
    encoding = _get_encoding(model)
    tokens = encoding.encode(text)

    chunks = []
    if len(tokens) == 0:
        return chunks

    step = chunk_size - chunk_overlap
    if step <= 0:
        step = 1

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break  # final window already consumed the tail; avoid duplicate stubs

    return chunks
