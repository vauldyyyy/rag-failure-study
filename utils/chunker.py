import tiktoken

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, model: str = "gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
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
    
    return chunks
