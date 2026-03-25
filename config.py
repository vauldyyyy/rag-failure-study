import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Chunking Config
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

# Retrieval Config
TOP_K = 3

# Models
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "models/gemini-2.5-flash"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "vector_store.index")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")
SQUAD_SUBSET_PATH = os.path.join(DATA_DIR, "squad_subset.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
