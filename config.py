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
# --------------------------------------------------------------------------- #
# Retrieval backend
# --------------------------------------------------------------------------- #
# "local"  = sentence-transformers on CPU. No API, no quota, fully deterministic,
#            and reproducible by anyone without a key — which matters for a paper.
# "gemini" = gemini-embedding-001 (3072-d). Stronger, but 1,330 chunks exceeds the
#            free-tier *daily* embedding cap, so the index cannot be built in one
#            day and every API call spent here is one not spent on generation.
#
# The retriever is a nuisance parameter in this study: perturbations are applied to
# the context, and the research question is about what the generator does given
# that context. A weaker retriever simply yields more items where the gold document
# is absent, which is the regime of interest.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_EMBED_DIM = 384

EMBED_MODEL = "models/gemini-embedding-001"

# Free-tier generation quota is capped per DAY *per model*
# (GenerateRequestsPerDayPerProjectPerModel-FreeTier), and the allowance varies by
# an order of magnitude between model ids: gemini-2.5-flash permits only 20/day,
# which cannot support a study needing thousands of calls. Run `python
# probe_models.py` to see what this key can actually reach today.
#
# Lite-class models are also the honest choice on the merits: they are what gets
# deployed for high-volume RAG, and the study is about high-volume RAG.
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-3.5-flash-lite")
# The judge must not be the same checkpoint as the generator, or faithfulness
# scores are self-evaluation — the single biggest threat to validity in this study.
# Defaulting to a different model family removes it, and because free-tier quota is
# capped per model, it also stops the judge competing with the generator for calls.
#
# It must also be a *lite* model. Observed free-tier daily caps: non-lite flash
# models (gemini-2.5-flash, gemini-3.6-flash) allow 20 requests/day, while lite
# models allow enough to run a study. A judge on a non-lite model blocks the run
# after 20 items no matter how much generator quota remains.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "models/gemini-3.1-flash-lite")
JUDGE_SELF_EVAL = JUDGE_MODEL == LLM_MODEL

# Scoring thresholds
F1_CORRECT_THRESHOLD = 0.5   # answerable item counts as correct at/above this token F1

# --------------------------------------------------------------------------- #
# Rate limits
# --------------------------------------------------------------------------- #
# Requests per minute, enforced client-side. Defaults are the Google AI free tier
# as reported by its own 429s; a batched embed call is billed per *item*, so these
# count items rather than HTTP calls. Raise them in .env on a paid key.
#
#   EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier = 100
#
EMBED_RPM = int(os.getenv("EMBED_RPM", "95"))    # slight headroom under 100
LLM_RPM = int(os.getenv("LLM_RPM", "9"))         # free-tier flash is ~10/min
# Per-day caps are the binding constraint on free tier and cannot be waited out.
# Set to 0 to disable the client-side stop and rely on server 429s alone.
LLM_RPD = int(os.getenv("LLM_RPD", "0"))

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
    """Tag index/cache paths by retrieval backend and dry-run state.

    Vectors from different embedders are not interchangeable — they have different
    dimensions and different geometry — so they must never share an index file.
    """
    root, ext = os.path.splitext(path)
    if DRY_RUN:
        return f"{root}.dryrun{ext}"
    if EMBED_BACKEND == "local":
        return f"{root}.local{ext}"
    return path


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
