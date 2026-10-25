"""Study configuration. Every experiment reads its settings from here.

Override the model choices with environment variables (or a .env file) rather than
editing this file, so results stay traceable to the settings recorded in each
summary.json.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# ---------------------------------------------------------------- reproducibility
SEED = 13

# ---------------------------------------------------------------- dataset
SQUAD_PATH_RAW = "data/squad_dev.json"      # SQuAD v2.0 dev set (committed)
N_EVAL_ARTICLES = 12        # articles whose paragraphs form the base corpus
N_ANSWERABLE = 300          # sampled answerable questions
N_UNANSWERABLE = 150        # sampled unanswerable (adversarial) questions

# ---------------------------------------------------------------- experiments
DEFAULT_K = 3
NOISE_LEVELS = [0, 100, 250, 500]           # off-topic distractor paragraphs added
K_VALUES = [1, 2, 3, 5, 7, 10]              # retrieved passages given to the model
POSITION_K = 10                             # context size for the position test
POSITIONS = [1, 3, 5, 7, 10]                # where the gold passage is placed
PROMPTS = ["open", "default", "strict"]     # see ragstudy/prompts.py

# ---------------------------------------------------------------- models
GEN_MODEL = os.getenv("RAG_GEN_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", GEN_MODEL)
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "768"))
# Gemini 2.5 Flash / Flash-Lite "think" by default. 0 turns thinking off so every
# condition is answered the same way; set RAG_THINKING_BUDGET=-1 to leave the
# model default (needed for 2.5 Pro, which cannot disable thinking).
THINKING_BUDGET = int(os.getenv("RAG_THINKING_BUDGET", "0"))
MAX_OUTPUT_TOKENS = 64
# Calls per minute ceiling (0 = no client-side throttle). Match your API tier.
REQUESTS_PER_MINUTE = int(os.getenv("RAG_RPM", "0"))

# OpenAI-compatible endpoint (Groq, OpenRouter, Ollama...) for --gen openai
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------- statistics
BOOTSTRAP_SAMPLES = 2000

# ---------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "processed", "dataset.json")
CACHE_PATH = os.path.join(BASE_DIR, "cache", "cache.sqlite")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
