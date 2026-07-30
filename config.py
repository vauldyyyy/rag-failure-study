import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Reproducibility. Every sampling / shuffling / noise-selection step seeds off this.
SEED = 20240617

# Chunking Config
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

# Retrieval Config
TOP_K = 3                 # default context size for the control condition
RETRIEVAL_DEPTH = 20      # depth cached once per query; every condition slices from this

# Evaluation set
N_QA = 600                # total questions sampled
FRAC_UNANSWERABLE = 0.30  # SQuAD v2 is_impossible share of the sample

# Models
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "models/gemini-2.5-flash"
# The judge must not be the same checkpoint as the generator, otherwise faithfulness
# scores are self-evaluation. Override via .env once a second provider is available;
# until then JUDGE_SELF_EVAL flags the limitation in every results file.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "models/gemini-2.5-flash")
JUDGE_SELF_EVAL = JUDGE_MODEL == LLM_MODEL

# Scoring thresholds
F1_CORRECT_THRESHOLD = 0.5   # answerable item counts as correct at/above this token F1

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "vector_store.index")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")
SQUAD_SUBSET_PATH = os.path.join(DATA_DIR, "squad_subset.json")

# Secondary dataset (multi-hop, different domain mix) — same schema as the SQuAD files.
HOTPOT_CORPUS_PATH = os.path.join(DATA_DIR, "hotpot_corpus.json")
HOTPOT_SUBSET_PATH = os.path.join(DATA_DIR, "hotpot_subset.json")
HOTPOT_INDEX_PATH = os.path.join(DATA_DIR, "hotpot_store.index")

QUERY_EMB_CACHE = os.path.join(CACHE_DIR, "query_embeddings.json")
RETRIEVAL_CACHE = os.path.join(CACHE_DIR, "retrieval_squad.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dry-run mode
# --------------------------------------------------------------------------- #
# Executes the whole pipeline against a deterministic stub model (utils/stub_llm.py)
# with no API key and no quota. Synthetic output is written to results_dryrun/ and
# uses separate index/cache files, so it can never be mixed with real results.
DRY_RUN = False


def set_dry_run(flag: bool):
    global DRY_RUN, RESULTS_DIR
    DRY_RUN = bool(flag)
    RESULTS_DIR = os.path.join(BASE_DIR, "results_dryrun" if DRY_RUN else "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _tag(path: str) -> str:
    """Insert a .dryrun marker before the extension when in dry-run mode."""
    if not DRY_RUN:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.dryrun{ext}"


def require_api_key():
    """Fail immediately rather than burning the retry/backoff ladder on 401s."""
    if DRY_RUN:
        return
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is not set.\n"
            "  cp .env.example .env   then paste a key from https://aistudio.google.com\n"
            "Only data/load_squad.py, data/load_hotpot.py, phase3 and the test suite "
            "run without one."
        )


def dataset_paths(name: str):
    """Return (corpus, qa_subset, index, retrieval_cache) for a dataset name."""
    if name == "squad":
        return (CORPUS_PATH, SQUAD_SUBSET_PATH, _tag(FAISS_INDEX_PATH),
                _tag(os.path.join(CACHE_DIR, "retrieval_squad.json")))
    if name == "hotpot":
        return (HOTPOT_CORPUS_PATH, HOTPOT_SUBSET_PATH, _tag(HOTPOT_INDEX_PATH),
                _tag(os.path.join(CACHE_DIR, "retrieval_hotpot.json")))
    raise ValueError(f"Unknown dataset: {name}")
