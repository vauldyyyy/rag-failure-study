import string
import re
import google.generativeai as genai
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

genai.configure(api_key=config.GEMINI_API_KEY)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        if not s: return ""
        return text.lower()
    if not s: return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    if not prediction or not ground_truth: return 0
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    if not prediction or not ground_truth: return 0
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def precision_at_k(retrieved_doc_ids, true_doc_id):
    return int(true_doc_id in retrieved_doc_ids)

def is_hallucinated(prediction):
    refusals = ["i cannot answer", "not contained in the context", "i don't know", "i do not know", "error generating"]
    pred_lower = prediction.lower()
    for ref in refusals:
        if ref in pred_lower:
            return 0 
            
    # Gemini Flash LLM as a judge for hallucination based purely on answer
    prompt = (
        f"You are a judge assessing if an Answer is a hallucination. "
        f"Answer: {prediction}\n\n"
        f"Respond 'HALLUCINATION' if it makes egregious unsupported factual claims, otherwise say 'FAITHFUL'."
    )
    import time
    for attempt in range(5):
        try:
            model = genai.GenerativeModel(config.LLM_MODEL)
            response = model.generate_content(prompt)
            text = response.text.lower()
            if "hallucination" in text:
                return 1
            return 0
        except Exception as e:
            if attempt == 4:
                return 0
            time.sleep(min(60, (2 ** attempt) * 4))
