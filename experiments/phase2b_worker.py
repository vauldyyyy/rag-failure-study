"""Run ONE Phase 2b condition, so the grid can be collected in parallel.

Phase 2b runs its eleven conditions one after another. On a slow generator (a
reasoning model behind a busy endpoint) that serialises hours of waiting. Each
condition is independent and checkpoints to its own file, so this worker runs a
single condition with exactly the arguments phase2b_noise_injection.py would use.
Afterwards, run phase2b_noise_injection.py with the same --model/--tag/--limit:
it finds every condition complete in its checkpoint, makes no API calls, and
computes the paired statistics and FDR correction as usual.

    python experiments/phase2b_worker.py clean --model M --tag T --limit 150
    python experiments/phase2b_worker.py related@50 --model M --tag T --limit 150

Build the retrieval cache once (--prepare) before starting workers in parallel,
so no two processes write it at the same time.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import phase2b_noise_injection as p2b  # GRID and K: single source
from utils import noise as noise_mod
from utils.pipeline import build_retrieval_cache, common_args, load_dataset, run_condition


def main():
    condition = sys.argv.pop(1)
    args = common_args(f"Phase 2b worker: {condition}")
    qa_pairs, corpus, store, cache_path = load_dataset(
        args.dataset, limit=args.limit, rebuild_index=args.rebuild_index)
    retrieval = build_retrieval_cache(qa_pairs, store, cache_path)
    if condition == "--prepare":
        print(f"Retrieval cache ready for {len(qa_pairs)} questions.")
        return

    grid = {f"{t}@{int(f * 100)}": (t, f) for t, f in p2b.GRID}
    if condition == "clean":
        noise_type, frac = noise_mod.CLEAN, 0.0
    elif condition in grid:
        noise_type, frac = grid[condition]
    else:
        sys.exit(f"Unknown condition {condition!r}; choose clean or one of {sorted(grid)}")

    rows, summary = run_condition(
        condition, qa_pairs, retrieval, corpus, k=p2b.K,
        noise_type=noise_type, noise_frac=frac,
        use_judge=not args.no_judge, model_name=args.model,
        repeats=args.repeats, temperature=args.temperature, dataset=args.dataset)
    valid = [r for r in rows if r["outcome"] != "INVALID"]
    print(f"{condition}: {len(rows)} rows, {len(rows) - len(valid)} invalid")


if __name__ == "__main__":
    main()
