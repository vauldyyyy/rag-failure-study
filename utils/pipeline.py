"""Shared experiment runner.

Every phase used to carry its own copy of the load/retrieve/generate/score loop.
That is not a style problem — it meant Phase 1 and Phase 4 were not guaranteed to
be running the same pipeline, so the difference between their numbers could have
come from the code rather than from the condition. One loop, one set of results.

Retrieval is executed once per question at RETRIEVAL_DEPTH and cached to disk.
Retrieval is deterministic given (query, index), so every condition slices the
same cached ranking: conditions differ only in how the context is assembled and
what prompt is used, never in what came back from FAISS.
"""

import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import noise as noise_mod
from utils.embedder import flush, get_embedding
from utils.llm import generate_answer
from utils.metrics import aggregate, precision_at_k, recall_at_k, score_item
from utils.vector_store import build_or_load


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_dataset(dataset="squad", limit=None, rebuild_index=False):
    config.require_api_key()
    corpus_path, qa_path, index_path, cache_path = config.dataset_paths(dataset)

    for path, hint in ((corpus_path, "loader"), (qa_path, "loader")):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing. Run data/load_{dataset}.py first ({hint})."
            )

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    if limit:
        # Take a prefix of the already-shuffled, stratified sample so a pilot run
        # keeps the answerable/unanswerable balance of the full set.
        qa_pairs = qa_pairs[:limit]

    store = build_or_load(corpus_path, index_path, rebuild=rebuild_index)
    return qa_pairs, corpus, store, cache_path


def build_retrieval_cache(qa_pairs, store, cache_path, depth=config.RETRIEVAL_DEPTH):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    missing = [qa for qa in qa_pairs if qa["qid"] not in cache]
    if missing:
        print(f"Retrieving top-{depth} for {len(missing)} queries...")
        for i, qa in enumerate(missing):
            emb = get_embedding(qa["question"], task_type="RETRIEVAL_QUERY")
            cache[qa["qid"]] = store.search(emb, top_k=depth)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(missing)}")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
        flush()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    return cache


# --------------------------------------------------------------------------- #
# Pre-generation signals (the "can we predict failure?" half of the question)
# --------------------------------------------------------------------------- #

def retrieval_signals(context, ranked, k):
    """Signals computable *before* the generator is called.

    Anything here is usable as a live abstention gate in a production system; the
    point of Phase 3 is to see whether any of them separate failures from successes.
    """
    scores = [c.get("score", 0.0) for c in context] or [0.0]
    tail = [c.get("score", 0.0) for c in ranked[:max(k, 2)]] or [0.0, 0.0]
    return {
        "top1_score": round(float(max(scores)), 4),
        "mean_score": round(float(sum(scores) / len(scores)), 4),
        "min_score": round(float(min(scores)), 4),
        "score_margin": round(float(tail[0] - tail[1]), 4) if len(tail) > 1 else 0.0,
        "score_spread": round(float(max(scores) - min(scores)), 4),
        "n_context": len(context),
    }


# --------------------------------------------------------------------------- #
# Condition runner
# --------------------------------------------------------------------------- #

def run_condition(name, qa_pairs, retrieval, corpus, k=config.TOP_K,
                  noise_type=noise_mod.CLEAN, noise_frac=0.0,
                  system_prompt=None, gold_position=None, use_judge=True,
                  rerank=None, seed=config.SEED, verbose=True,
                  model_name=config.LLM_MODEL, repeats=1, temperature=0.0):
    """Run one experimental cell and return (rows, summary).

    repeats > 1 re-runs every item, which only produces variation at
    temperature > 0. Rows carry a `repeat` index; `stats.collapse_repeats` averages
    within an item before any interval is computed, so repeated sampling never
    inflates n.
    """
    rows = []

    for r in range(repeats):
        # Re-seeded per repeat so distractor draws are reproducible but not
        # identical across repeats.
        rng = random.Random(seed + r)

        for i, qa in enumerate(qa_pairs):
            ranked = retrieval.get(qa["qid"], [])

            context = noise_mod.apply_noise(
                ranked, qa, k,
                noise_type=noise_type, noise_frac=noise_frac,
                corpus=corpus, rng=rng, gold_position=gold_position,
            )
            if rerank is not None:
                context = rerank(qa["question"], context)

            if config.DRY_RUN:
                from utils.stub_llm import stub_generate
                gen = stub_generate(
                    qa["question"], context, system_prompt=system_prompt,
                    gold_answers=qa.get("answers"),
                    gold_doc_ids=qa.get("gold_doc_ids") or [qa.get("gold_doc_id")],
                    is_answerable=qa.get("is_answerable", True),
                )
            else:
                gen = generate_answer(qa["question"], context,
                                      model_name=model_name,
                                      system_prompt=system_prompt,
                                      temperature=temperature)

            row = score_item(qa, gen["text"], gen["status"], context, use_judge=use_judge)

            ctx_doc_ids = [c["doc_id"] for c in context]
            gold_ids = qa.get("gold_doc_ids") or [qa.get("gold_doc_id")]
            row["p_at_k"] = precision_at_k(ctx_doc_ids, qa.get("gold_doc_id"))
            row["recall_at_k"] = round(recall_at_k(ctx_doc_ids, gold_ids), 4)
            row["signals"] = retrieval_signals(context, ranked, k)
            row["condition"] = name
            row["repeat"] = r
            # Audit trail for the perturbation itself. COUNTERFACTUAL in particular
            # can silently no-op when the gold string is absent from the chunk, and
            # DROP_GOLD is a no-op on items whose gold was never retrieved — either
            # dilutes the condition without moving any headline number.
            row["n_perturbed"] = sum(1 for c in context if c.get("perturbed"))
            if noise_type == noise_mod.DROP_GOLD:
                gold_set = {g for g in gold_ids if g}
                row["treatment_applied"] = any(
                    c["doc_id"] in gold_set for c in ranked[:k])
            else:
                row["treatment_applied"] = row["n_perturbed"] > 0
            rows.append(row)

            if verbose and (i + 1) % 50 == 0:
                suffix = f" (repeat {r + 1}/{repeats})" if repeats > 1 else ""
                print(f"  [{name}] {i + 1}/{len(qa_pairs)}{suffix}")

    summary = aggregate(rows)
    if noise_type in (noise_mod.CLEAN, noise_mod.DROP_GOLD):
        n_expected = 0  # DROP_GOLD removes slots rather than substituting into them
    elif noise_type == noise_mod.COUNTERFACTUAL:
        n_expected = max(1, int(round(k * noise_frac)))
    else:
        n_expected = int(round(k * noise_frac))
    summary.update({
        "treatment_applied_rate": round(
            sum(1 for r in rows if r["treatment_applied"]) / len(rows), 4) if rows else None,
        "mean_slots_perturbed": round(
            sum(r["n_perturbed"] for r in rows) / len(rows), 3) if rows else None,
        "slots_perturbed_expected": n_expected,
        "condition": name, "k": k, "noise_type": noise_type,
        "noise_frac": noise_frac, "gold_position": gold_position,
        "seed": seed, "model": "stub" if config.DRY_RUN else model_name,
        "repeats": repeats, "temperature": temperature,
        "synthetic": config.DRY_RUN,
    })
    return rows, summary


def save_results(filename, payload):
    path = os.path.join(config.RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}")
    return path


def run_manifest(dataset, qa_pairs, extra=None):
    """Provenance stamped into every results file."""
    manifest = {
        "synthetic": config.DRY_RUN,
        "dataset": dataset,
        "n_qa": len(qa_pairs),
        "n_answerable": sum(1 for q in qa_pairs if q.get("is_answerable", True)),
        "n_unanswerable": sum(1 for q in qa_pairs if not q.get("is_answerable", True)),
        "llm_model": config.LLM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "judge_self_eval": config.JUDGE_SELF_EVAL,
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_depth": config.RETRIEVAL_DEPTH,
        "f1_correct_threshold": config.F1_CORRECT_THRESHOLD,
        "seed": config.SEED,
        "temperature": 0.0,
    }
    manifest.update(extra or {})
    return manifest


def common_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", default="squad", choices=["squad", "hotpot"])
    p.add_argument("--limit", type=int, default=None,
                   help="Run on the first N questions (pilot runs on free-tier quota)")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip LLM-judge calls; rule-based labels only")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the whole pipeline against a deterministic stub model. "
                        "No API key, no quota. Output goes to results_dryrun/ and is "
                        "stamped synthetic — it verifies plumbing, not findings.")
    p.add_argument("--model", default=config.LLM_MODEL,
                   help="Generator model id; vary it to test cross-model generality")
    p.add_argument("--repeats", type=int, default=1,
                   help="Samples per item. Only meaningful with --temperature > 0")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    config.set_dry_run(args.dry_run)
    if args.dry_run:
        print("*** DRY RUN — stub model, synthetic numbers, results_dryrun/ ***\n")
    if args.repeats > 1 and args.temperature == 0.0:
        print("Warning: --repeats > 1 at temperature 0 produces identical samples.\n")
    return args


def announce_budget(n_conditions, n_items, use_judge=True, repeats=1):
    if config.DRY_RUN:
        print(f"\nAPI budget: 0 (dry run) — {n_conditions} conditions x {n_items} items"
              f"{f' x {repeats} repeats' if repeats > 1 else ''}.\n")
        return
    gen = n_conditions * n_items * repeats
    judge = int(gen * 0.4) if use_judge else 0
    print(f"\nAPI budget: ~{gen} generation calls"
          f"{f' + up to ~{judge} judge calls' if judge else ''} "
          f"({n_conditions} conditions x {n_items} items"
          f"{f' x {repeats} repeats' if repeats > 1 else ''}).")
    print("At the free-tier 15 RPM this is roughly "
          f"{(gen + judge) / 15 / 60:.1f} h. Use --limit for a pilot.\n")
