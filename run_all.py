"""Run the full study in dependency order.

    python run_all.py --limit 150      # pilot: fits comfortably in free-tier quota
    python run_all.py                  # full 600-item run

Phases share the retrieval cache and the query-embedding cache, so re-running a
phase after a crash does not re-pay for retrieval.
"""

import argparse
import subprocess
import sys
import os

STEPS = [
    ("Phase 1  baseline", ["experiments/phase1_baseline.py"]),
    ("Phase 2b noise sweep", ["experiments/phase2b_noise_injection.py"]),
    ("Phase 2c context size", ["experiments/phase2c_context_size.py"]),
    ("Phase 4  mitigations", ["experiments/phase4_fixes.py"]),
    ("Phase 3  predictability", ["experiments/phase3_predictability.py"]),  # analysis only
    ("Figures", ["visualize_results.py"]),
]


def main():
    p = argparse.ArgumentParser(description="Run every phase in order")
    p.add_argument("--dataset", default="squad", choices=["squad", "hotpot"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Stub model, no API key, no quota; output to results_dryrun/")
    p.add_argument("--model", default=None)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tag", default="", help="Suffix results/checkpoints (cross-model runs)")
    p.add_argument("--from-step", type=int, default=0, help="Skip the first N steps")
    args = p.parse_args()

    for i, (label, cmd) in enumerate(STEPS):
        if i < args.from_step:
            print(f"[skip] {label}")
            continue

        full = [sys.executable] + cmd + ["--dataset", args.dataset]
        if args.dry_run:
            full += ["--dry-run"]
        # Phase 3 and the figure script only take --dataset and --dry-run.
        if "phase3" not in cmd[0] and "visualize" not in cmd[0]:
            if args.limit:
                full += ["--limit", str(args.limit)]
            if args.no_judge:
                full += ["--no-judge"]
            if args.model:
                full += ["--model", args.model]
            if args.repeats > 1:
                full += ["--repeats", str(args.repeats)]
            if args.temperature:
                full += ["--temperature", str(args.temperature)]
            if args.tag:
                full += ["--tag", args.tag]

        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        result = subprocess.run(full, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode == 3:
            # Daily quota exhausted. Checkpoints are on disk; this is not a failure.
            print(f"\n{label} stopped on daily quota. Progress is checkpointed.")
            print(f"Resume tomorrow with:  python run_all.py --from-step {i}"
                  + (f" --limit {args.limit}" if args.limit else ""))
            sys.exit(0)
        if result.returncode != 0:
            print(f"\n{label} failed (exit {result.returncode}). "
                  f"Fix, then resume with --from-step {i}")
            sys.exit(result.returncode)

    print("\nAll phases complete. Results and figures are in results/.")


if __name__ == "__main__":
    main()
