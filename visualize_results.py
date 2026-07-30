"""Figure generation.

Figures plot rates with their bootstrap intervals. A decay curve without error
bars invites the reader to see a trend in four points that a resample would erase,
which is the main way small-n RAG studies overclaim.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

def _out(name):
    """Resolved at call time — config.RESULTS_DIR moves under --dry-run."""
    return os.path.join(config.RESULTS_DIR, name)


def _stamp(fig):
    """Make synthetic figures impossible to mistake for results."""
    if config.DRY_RUN:
        fig.text(0.5, 0.5, "SYNTHETIC\nDRY RUN", fontsize=44, color="red",
                 alpha=0.18, ha="center", va="center", rotation=30,
                 transform=fig.transFigure, zorder=10)


def _load(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _errbars(cis):
    """Convert [{mean,lo,hi}] into the asymmetric yerr matplotlib expects."""
    means = [c["mean"] for c in cis]
    lo = [max(0.0, m - c["lo"]) for m, c in zip(means, cis)]
    hi = [max(0.0, c["hi"] - m) for m, c in zip(means, cis)]
    return means, [lo, hi]


def fig_noise_curve(dataset):
    data = _load(f"phase2b_noise_{dataset}.json")
    if not data:
        return
    conditions = data["conditions"]

    fracs = [0.25, 0.5, 0.75, 1.0]
    names = [f"related@{int(f * 100)}" for f in fracs]
    present = [(f, n) for f, n in zip(fracs, names) if n in conditions]
    if not present:
        return

    x = [int(f * 100) for f, _ in present]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for key, label, color in (
        ("correct_rate", "Correct", "#2a9d8f"),
        ("abstention_rate", "Abstained (safe failure)", "#457b9d"),
        ("hallucination_rate", "Unsupported assertion (silent failure)", "#e63946"),
    ):
        cis = [conditions[n]["ci"][key] for _, n in present if key in conditions[n].get("ci", {})]
        if len(cis) != len(present):
            continue
        means, yerr = _errbars(cis)
        ax.errorbar(x, means, yerr=yerr, marker="o", capsize=3, label=label, color=color)

    ax.set_xlabel("Share of context slots replaced by related-but-wrong chunks (%)")
    ax.set_ylabel("Rate")
    ax.set_title("Outcome composition as retrieval degrades\n(error bars: 95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _stamp(fig)
    out = _out(f"fig1_noise_curve_{dataset}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig_context_size(dataset):
    data = _load(f"phase2c_context_{dataset}.json")
    if not data:
        return

    size = data.get("size_conditions", {})
    ks, f1_cis, gold = [], [], []
    for name, summary in size.items():
        if not name.startswith("k="):
            continue
        ks.append(int(name.split("=")[1]))
        f1_cis.append(summary["ci"]["F1"])
        gold.append(summary.get("gold_in_context_rate"))
    if not ks:
        return

    order = sorted(range(len(ks)), key=lambda i: ks[i])
    ks = [ks[i] for i in order]
    f1_cis = [f1_cis[i] for i in order]
    gold = [gold[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    means, yerr = _errbars(f1_cis)
    ax.errorbar(ks, means, yerr=yerr, marker="o", capsize=3, color="#2a9d8f", label="Answer F1")
    ax.plot(ks, gold, marker="s", linestyle="--", color="#8d99ae", label="Gold doc in context")
    ax.set_xlabel("Context size K (chunks passed to the generator)")
    ax.set_ylabel("Rate")
    ax.set_title("More retrieved context is not monotonically better\n(error bars: 95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _stamp(fig)
    out = _out(f"fig2_context_size_{dataset}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")

    pos = data.get("position_conditions", {})
    if pos:
        labels = list(pos.keys())
        cis = [pos[l]["ci"]["F1"] for l in labels]
        means, yerr = _errbars(cis)
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.bar(labels, means, yerr=yerr, capsize=4, color="#457b9d")
        ax.set_ylabel("Answer F1")
        ax.set_xlabel(f"Slot the gold chunk was placed in (K={data['manifest'].get('position_k')})")
        ax.set_title("Gold position within a fixed context")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        _stamp(fig)
        out = _out(f"fig3_gold_position_{dataset}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")


def fig_mitigations(dataset):
    data = _load(f"phase4_fixes_{dataset}.json")
    if not data:
        return
    conditions = data["conditions"]

    names, d_hall, d_corr = [], [], []
    for name, summary in conditions.items():
        vs = summary.get("vs_none")
        if not vs or not vs.get("hallucination_rate") or not vs.get("correct_rate"):
            continue
        names.append(name)
        d_hall.append(vs["hallucination_rate"]["diff"])
        d_corr.append(vs["correct_rate"]["diff"])
    if not names:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.axhline(0, color="#adb5bd", lw=1)
    ax.axvline(0, color="#adb5bd", lw=1)
    # Break-even line: below it, a mitigation costs more accuracy than it saves.
    lim = max(abs(min(d_hall + d_corr)), abs(max(d_hall + d_corr))) * 1.3 or 0.1
    ax.plot([-lim, lim], [-lim, lim], ls=":", color="#adb5bd", lw=1, label="break-even")

    ax.scatter(d_hall, d_corr, s=90, color="#e63946", zorder=3)
    for n, x, y in zip(names, d_hall, d_corr):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)

    ax.set_xlabel("Δ hallucination rate vs no mitigation (negative = better)")
    ax.set_ylabel("Δ correct rate vs no mitigation (negative = worse)")
    ax.set_title("Mitigations trade accuracy for safety\nUpper-left quadrant is the only real win")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _stamp(fig)
    out = _out(f"fig4_mitigation_tradeoff_{dataset}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description="Generate figures from results/")
    p.add_argument("--dataset", default="squad", choices=["squad", "hotpot"])
    p.add_argument("--dry-run", action="store_true",
                   help="Plot results_dryrun/; figures are watermarked SYNTHETIC")
    args = p.parse_args()
    config.set_dry_run(args.dry_run)

    produced_any = False
    for fn in (fig_noise_curve, fig_context_size, fig_mitigations):
        before = len(os.listdir(config.RESULTS_DIR))
        fn(args.dataset)
        produced_any |= len(os.listdir(config.RESULTS_DIR)) > before

    if not produced_any:
        print(f"No result files found for '{args.dataset}' in {config.RESULTS_DIR}. "
              "Run the experiment phases first.")


if __name__ == "__main__":
    main()
