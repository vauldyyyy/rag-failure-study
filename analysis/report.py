"""Turn results/<model>/<exp>/ into tables, paired significance tests and figures.

Writes results/<model>/RESULTS.md and results/<model>/figures/*.png.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from ragstudy import metrics  # noqa: E402

# First three slots of the validated categorical palette (pass all-pairs CVD checks).
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False, "figure.dpi": 150,
})


def _load(model_dir, exp):
    path = os.path.join(model_dir, exp, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _records(model_dir, exp, cond):
    path = os.path.join(model_dir, exp, f"{cond}.jsonl")
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def _fmt(m, key, pct=True):
    if key not in m:
        return "–"
    v = m[key]
    s = 100 if pct else 1
    return f"{v['mean'] * s:.1f} [{v['lo'] * s:.1f}, {v['hi'] * s:.1f}]"


def _table(summary, cols):
    head = "| condition | " + " | ".join(c for c, _ in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [f"| {name} | " + " | ".join(_fmt(c['metrics'], key) for _, key in cols) + " |"
            for name, c in summary["conditions"].items()]
    return "\n".join([head, sep] + rows)


def _paired(model_dir, exp, a, b, field, impossible):
    ra, rb = _records(model_dir, exp, a), _records(model_dir, exp, b)
    qids = [q for q in ra if q in rb and ra[q]["is_impossible"] == impossible
            and ra[q].get(field) is not None and rb[q].get(field) is not None]
    if not qids:
        return None
    t = metrics.paired_bootstrap([float(ra[q][field]) for q in qids],
                                 [float(rb[q][field]) for q in qids])
    return (f"{exp}: {b} vs {a} — {field}: {t['diff'] * 100:+.1f} pts "
            f"[{t['lo'] * 100:+.1f}, {t['hi'] * 100:+.1f}], p={t['p']:.3f}, n={t['n']}")


def _line(ax, xs, summ, names, key, label, color):
    ys = [summ["conditions"][n]["metrics"].get(key) for n in names]
    pts = [(x, y) for x, y in zip(xs, ys) if y]
    if not pts:
        return
    x = [p[0] for p in pts]
    ax.plot(x, [p[1]["mean"] * 100 for p in pts], marker="o", color=color, label=label)
    ax.fill_between(x, [p[1]["lo"] * 100 for p in pts], [p[1]["hi"] * 100 for p in pts],
                    color=color, alpha=0.15, linewidth=0)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return os.path.basename(path)


def report_model(model_dir):
    name = os.path.basename(model_dir)
    fig_dir = os.path.join(model_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    md = [f"# Results — {name}", ""]
    tests, figs = [], []

    s = _load(model_dir, "E1")
    if s:
        md += ["## E1 Baseline (K=3, no noise, default prompt)", "",
               "Values are % with 95% bootstrap CIs.", "",
               _table(s, [("EM", "em"), ("F1", "f1"), ("hallucination (unans.)", "hallucination_rate"),
                          ("over-refusal (ans.)", "over_refusal"), ("retrieval hit@3", "retrieval_hit"),
                          ("faithfulness", "faithfulness")]), ""]
        m = s["meta"]
        md += [f"Generator `{m['generator']}`, embedder `{m['embedder']}`, judge `{m['judge']}`, "
               f"n={m['n_questions']}, git `{m['git_sha']}`.", ""]

    s = _load(model_dir, "E2")
    if s:
        names = list(s["conditions"])
        xs = [s["conditions"][n]["settings"]["noise"] for n in names]
        md += ["## E2 Retrieval noise (off-topic distractor paragraphs)", "",
               _table(s, [("retrieval hit@3", "retrieval_hit"), ("MRR", "mrr"), ("EM", "em"),
                          ("F1", "f1"), ("hallucination", "hallucination_rate")]), ""]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        _line(ax, xs, s, names, "retrieval_hit", "Retrieval hit@3", C1)
        _line(ax, xs, s, names, "f1", "F1 (answerable)", C2)
        _line(ax, xs, s, names, "hallucination_rate", "Hallucination (unanswerable)", C3)
        ax.set_xlabel("Distractor paragraphs added to the index")
        ax.set_ylabel("%")
        ax.legend()
        figs.append(_save(fig, os.path.join(fig_dir, "e2_noise.png")))
        tests += [_paired(model_dir, "E2", names[0], names[-1], f, imp)
                  for f, imp in [("retrieval_hit", False), ("f1", False), ("hallucinated", True)]]

    s = _load(model_dir, "E2b")
    if s:
        md += ["## E2b Conflicting evidence (counterfactual paragraph in the index)", "",
               "Misled rate = the model returned the planted counterfactual answer, among questions "
               "whose retrieved context contained the counterfactual paragraph.", "",
               _table(s, [("EM", "em"), ("F1", "f1"), ("misled", "misled_rate"),
                          ("over-refusal", "over_refusal")]), ""]

    s = _load(model_dir, "E3")
    if s:
        names = list(s["conditions"])
        xs = [s["conditions"][n]["settings"]["k"] for n in names]
        md += ["## E3 Context size", "",
               _table(s, [("retrieval hit@K", "retrieval_hit"), ("EM", "em"), ("F1", "f1"),
                          ("hallucination", "hallucination_rate"), ("over-refusal", "over_refusal")]), ""]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        _line(ax, xs, s, names, "retrieval_hit", "Retrieval hit@K", C1)
        _line(ax, xs, s, names, "f1", "F1 (answerable)", C2)
        _line(ax, xs, s, names, "hallucination_rate", "Hallucination (unanswerable)", C3)
        ax.set_xlabel("Passages in context (K)")
        ax.set_ylabel("%")
        ax.set_xticks(xs)
        ax.legend()
        figs.append(_save(fig, os.path.join(fig_dir, "e3_context_size.png")))
        tests += [_paired(model_dir, "E3", names[0], names[-1], "f1", False),
                  _paired(model_dir, "E3", "k_3", names[-1], "hallucinated", True)]

    s = _load(model_dir, "E4")
    if s:
        names = list(s["conditions"])
        xs = [s["conditions"][n]["settings"]["position"] for n in names]
        md += ["## E4 Position of the gold passage (10 passages)", "",
               _table(s, [("EM", "em"), ("F1", "f1"), ("over-refusal", "over_refusal")]), ""]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        _line(ax, xs, s, names, "f1", "F1", C2)
        _line(ax, xs, s, names, "em", "Exact match", C1)
        ax.set_xlabel("Position of the gold passage (1 = first)")
        ax.set_ylabel("%")
        ax.set_xticks(xs)
        ax.legend()
        figs.append(_save(fig, os.path.join(fig_dir, "e4_position.png")))
        mid = names[len(names) // 2]
        tests += [_paired(model_dir, "E4", names[0], mid, "f1", False),
                  _paired(model_dir, "E4", names[-1], mid, "f1", False)]

    s = _load(model_dir, "E5")
    if s:
        md += ["## E5 Prompting strategy", "",
               "The trade-off: lower hallucination on unanswerable questions vs more refusals on "
               "answerable ones.", "",
               _table(s, [("hallucination", "hallucination_rate"), ("over-refusal", "over_refusal"),
                          ("EM", "em"), ("F1", "f1"), ("SQuAD-v2 EM", "v2_em")]), ""]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        colors = {"open": C1, "default": C2, "strict": C3}
        for n, c in s["conditions"].items():
            m = c["metrics"]
            if "hallucination_rate" in m and "over_refusal" in m:
                st = c["settings"]
                x, y = m["over_refusal"]["mean"] * 100, m["hallucination_rate"]["mean"] * 100
                ax.scatter(x, y, s=60, color=colors.get(st["prompt"], MUTED),
                           marker="o" if st["noise"] == 0 else "s",
                           edgecolor="white", linewidth=1.5, zorder=3)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=c, marker="o", linestyle="", label=f"{p} prompt")
                   for p, c in colors.items()]
        handles += [Line2D([], [], color=MUTED, marker="o", linestyle="", label="no noise"),
                    Line2D([], [], color=MUTED, marker="s", linestyle="",
                           label=f"{max(config.NOISE_LEVELS)} distractors")]
        ax.legend(handles=handles, fontsize=8, loc="best")
        ax.set_xlabel("Over-refusal on answerable questions (%)")
        ax.set_ylabel("Hallucination on unanswerable (%)")
        figs.append(_save(fig, os.path.join(fig_dir, "e5_prompting.png")))
        if "default_noise_0" in s["conditions"] and "strict_noise_0" in s["conditions"]:
            tests += [_paired(model_dir, "E5", "default_noise_0", "strict_noise_0", "hallucinated", True),
                      _paired(model_dir, "E5", "default_noise_0", "strict_noise_0", "over_refusal", False),
                      _paired(model_dir, "E5", "open_noise_0", "default_noise_0", "hallucinated", True)]

    tests = [t for t in tests if t]
    if tests:
        md += ["## Paired comparisons (bootstrap, same questions)", ""] + [f"- {t}" for t in tests] + [""]
    if figs:
        md += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figs] + [""]
    out = os.path.join(model_dir, "RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return out


def build_report(model=None, results_dir=config.RESULTS_DIR):
    dirs = [os.path.join(results_dir, model)] if model else sorted(
        d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d))
    if not dirs:
        return "No results yet. Run: python run.py run all"
    return "\n".join(f"Wrote {report_model(d)}" for d in dirs)
