"""What sample size does each hypothesis actually need? Run this BEFORE spending quota.

Answering "how many questions should I run" after the run is how studies end up
with four points, wide intervals, and no conclusion. This script costs nothing and
takes seconds, and it exposes an asymmetry that should drive the run strategy:

  * H4 (Phase 3, AUROC) pools rows across every condition of every phase, so it is
    well estimated even from a small pilot.
  * H1 (Phase 2b, non-monotonic silent-failure curve) needs a *per-condition* rate
    at each noise level, and those intervals shrink only with the item count.

So a pilot can settle the paper's headline claim while leaving its supporting claim
undetectable. If quota is tight, spend it on Phase 2b rather than spreading it.

    python power_analysis.py
    python power_analysis.py --n 150 --n 300 --n 600
"""

import argparse

from utils.stats import ci_halfwidth, mde_paired_rate, n_for_halfwidth

# Conditions per phase, used to work out how many rows Phase 3 gets to pool.
CONDITIONS = {"phase1": 1, "phase2b": 10, "phase2c": 11, "phase4": 5}
TOTAL_CONDITIONS = sum(CONDITIONS.values())

FRAC_UNANSWERABLE = 0.30


def report(n_values, target_halfwidth, discordance):
    print(f"Assumptions: rates near 0.5 (worst case for width), "
          f"95% intervals, 80% power,\n"
          f"discordant-pair share {discordance} for paired tests, "
          f"{int(FRAC_UNANSWERABLE * 100)}% of items unanswerable.\n")

    header = f"{'n':>6}{'rate ±':>10}{'unans ±':>11}{'paired MDE':>13}{'phase3 rows':>14}"
    print(header)
    print("-" * len(header))

    for n in n_values:
        n_unans = int(round(n * FRAC_UNANSWERABLE))
        rate_hw = ci_halfwidth(0.5, n)
        unans_hw = ci_halfwidth(0.5, n_unans)
        mde = mde_paired_rate(n, discordance=discordance)
        rows = n * TOTAL_CONDITIONS
        print(f"{n:>6}{rate_hw:>10.3f}{unans_hw:>11.3f}{mde:>13.3f}{rows:>14,}")

    print(f"\nrate ±       half-width of a per-condition rate (the Table 1 cells)")
    print(f"unans ±      half-width on the {int(FRAC_UNANSWERABLE * 100)}% unanswerable subset "
          f"— the hallucination-rate column")
    print(f"paired MDE   smallest condition-vs-control difference detectable at 80% power")
    print(f"phase3 rows  generations pooled for the AUROC estimate "
          f"({TOTAL_CONDITIONS} conditions x n)")

    print("\n--- What each hypothesis needs ---\n")

    need = n_for_halfwidth(0.5, target_halfwidth)
    print(f"H1 (non-monotonic silent-failure curve)")
    print(f"  The interior peak must clear both endpoints' intervals. That means the")
    print(f"  peak has to exceed them by roughly 2x the half-width.")
    for n in n_values:
        hw = ci_halfwidth(0.5, n)
        print(f"    n={n:<5} peak must beat endpoints by >{2 * hw * 100:.0f} percentage points")
    print(f"  For ±{target_halfwidth:.2f} per-condition precision you need n≈{need}.")
    print(f"  VERDICT: underpowered below n≈400 unless the effect is large.\n")

    print(f"H2 (related vs random abstention, paired)")
    for n in n_values:
        print(f"    n={n:<5} detects a difference of >{mde_paired_rate(n, discordance) * 100:.0f} pp")
    print()

    print(f"H4 (retrieval signals predict silent failure, AUROC)")
    for n in n_values:
        rows = n * TOTAL_CONDITIONS
        # AUROC standard error is driven by the smaller class; assume ~10% positives.
        pos = max(rows * 0.10, 1)
        se = (0.25 / pos) ** 0.5  # conservative Hanley-McNeil style approximation
        print(f"    n={n:<5} {rows:>6,} pooled rows, ~{int(pos):,} positives, "
              f"AUROC SE ≈ {se:.3f}")
    print(f"  VERDICT: well powered even at n=150 — Phase 3 pools every condition.\n")

    print("--- Recommendation ---")
    print("If quota forces a choice, run Phase 2b at the largest n you can afford and")
    print("trim Phase 2c instead. Phase 3 inherits rows from whatever else you run, so")
    print("it costs nothing extra and stays reliable. Reporting H1 as 'underpowered at")
    print("this n' is a legitimate result; reporting it from four wide intervals is not.")


def main():
    p = argparse.ArgumentParser(description="Sample-size planning — no API calls")
    p.add_argument("--n", type=int, action="append", dest="n_values",
                   help="Sample size to evaluate (repeatable)")
    p.add_argument("--halfwidth", type=float, default=0.05,
                   help="Target half-width for a per-condition rate")
    p.add_argument("--discordance", type=float, default=0.25,
                   help="Expected share of discordant pairs; replace with the "
                        "observed value after the pilot")
    args = p.parse_args()

    report(args.n_values or [100, 150, 300, 600, 1000],
           args.halfwidth, args.discordance)


if __name__ == "__main__":
    main()
