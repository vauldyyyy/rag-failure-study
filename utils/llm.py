import google.generativeai as genai
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

genai.configure(api_key=config.GEMINI_API_KEY)

def generate_answer(query: str, retrieved_contexts: list, model_name: str = config.LLM_MODEL, system_prompt: str = None):
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Use the following context to answer the question. "
            "If the answer is not contained in the context, say 'I cannot answer based on the context'."
        )
        
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt
    )
        
    context_text = "\n\n".join([f"Context {i+1}:\n{c['chunk']}" for i, c in enumerate(retrieved_contexts)])
    
    user_prompt = (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{context_text}\n"
        f"---------------------\n"
        f"Given the context information and not prior knowledge, answer the query.\n"
        f"Query: {query}\n"
        f"Answer: "
    )
    
    import time
    for attempt in range(5):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            return response.text
        except Exception as e:
            if attempt == 4:
                print(f"LLM Error: {e}")
                return "Error generating response"
            
            # Google API Free Tier is 15 RPM.
            # If we hit this, we sleep exponentially, capping at around 60 seconds
            sleep_time = min(60, (2 ** attempt) * 4) 
            time.sleep(sleep_time)
