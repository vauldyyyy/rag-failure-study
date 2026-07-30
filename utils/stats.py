"""Uncertainty quantification.

Every rate in this study is a proportion over 600 (often far fewer, once split by
answerable/unanswerable) items. A 4-point difference between two conditions at
n=180 is inside the noise floor, so no rate is reported without an interval and no
comparison is claimed without a paired test on the same questions.
"""

import numpy as np


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 4),
        "lo": round(float(np.percentile(means, 100 * alpha / 2)), 4),
        "hi": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4),
        "n": int(arr.size),
    }


def paired_bootstrap_diff(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Paired bootstrap over items for mean(b) - mean(a).

    a and b must be aligned item-wise (same questions, same order); pairing is what
    removes per-question difficulty from the comparison.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Unpaired inputs: {a_arr.shape} vs {b_arr.shape}")
    if a_arr.size == 0:
        return {"diff": None, "lo": None, "hi": None, "p_two_sided": None, "n": 0}

    diff = b_arr - a_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)

    observed = float(diff.mean())
    # Two-sided bootstrap p: how often does a resample land on the other side of 0.
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "diff": round(observed, 4),
        "lo": round(float(np.percentile(boot, 100 * alpha / 2)), 4),
        "hi": round(float(np.percentile(boot, 100 * (1 - alpha / 2))), 4),
        "p_two_sided": round(float(min(p, 1.0)), 4),
        "n": int(diff.size),
    }


def auroc(scores, labels):
    """AUROC via the Mann-Whitney U identity, with tie correction.

    labels: 1 = positive class (e.g. the generation failed), 0 = negative.
    Returns None when one class is absent.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)

    # Average ranks within tied score groups.
    sorted_s = s[order]
    i = 0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return round(float(u / (n_pos * n_neg)), 4)


def rate_with_ci(rows, predicate, seed=0):
    """Bootstrap CI for the share of `rows` satisfying `predicate`."""
    return bootstrap_ci([1.0 if predicate(r) else 0.0 for r in rows], seed=seed)


def collapse_repeats(rows, value_fn):
    """Per-item mean of `value_fn` when a condition was sampled more than once.

    `value_fn` may return a bool (→ a rate) or a number (→ a mean such as F1).

    Repeated samples of the same question are not independent observations. Feeding
    them to the bootstrap as separate rows would shrink the interval by a factor of
    sqrt(repeats) and manufacture significance. Collapsing to one value per question
    keeps n equal to the number of questions, which is what it actually is.
    """
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r.get("qid"), []).append(float(value_fn(r) or 0.0))
    return [sum(v) / len(v) for v in by_qid.values()]


def rate_with_ci_clustered(rows, value_fn, seed=0):
    """rate_with_ci, but safe when rows contain repeated samples per question.

    Identical to rate_with_ci when repeats == 1, so it is the safe default.
    """
    return bootstrap_ci(collapse_repeats(rows, value_fn), seed=seed)


def benjamini_hochberg(pvalues, alpha=0.05):
    """FDR correction over a family of tests.

    This study runs ~25 paired comparisons against a single control. At alpha=0.05
    that is roughly one false positive expected *by construction*, so an uncorrected
    "significant" cell in Table 1 means very little. BH controls the expected share
    of false discoveries among the rejected hypotheses, which is the right error
    rate for a screening table like this one (Bonferroni would be correct but far
    too conservative across a correlated dose-response family).

    Returns a list of dicts aligned with the input, each with the adjusted q-value
    and whether it survives at `alpha`. None p-values pass through as None.
    """
    indexed = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    out = [None] * len(pvalues)
    if not indexed:
        return out

    indexed.sort(key=lambda x: x[1])
    m = len(indexed)

    # Step-up: walk from the largest p down, keeping a running minimum so the
    # adjusted values are monotone.
    running_min = 1.0
    adjusted = {}
    for rank in range(m, 0, -1):
        i, p = indexed[rank - 1]
        running_min = min(running_min, p * m / rank)
        adjusted[i] = min(running_min, 1.0)

    for i, q in adjusted.items():
        out[i] = {"q_value": round(q, 4), "significant_fdr": q <= alpha}
    return out


def apply_fdr(comparisons, alpha=0.05):
    """Annotate a {name: paired_bootstrap_diff result} mapping in place with FDR.

    Comparisons whose interval excluded zero but whose q-value does not survive
    are exactly the ones that would have been over-claimed.
    """
    names = list(comparisons)
    pvals = [(comparisons[n] or {}).get("p_two_sided") for n in names]
    for name, adj in zip(names, benjamini_hochberg(pvals, alpha)):
        if comparisons[name] is not None and adj is not None:
            comparisons[name].update(adj)
    return comparisons


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #

_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}
_Z_ALPHA_TWO_SIDED = 1.9600  # alpha = 0.05


def ci_halfwidth(p, n, z=_Z_ALPHA_TWO_SIDED):
    """Normal-approximation half-width of a 95% interval for a rate."""
    if n <= 0:
        return None
    return z * (p * (1 - p) / n) ** 0.5


def n_for_halfwidth(p, halfwidth, z=_Z_ALPHA_TWO_SIDED):
    """Items needed for a rate near `p` to be estimated to +/- `halfwidth`."""
    if halfwidth <= 0:
        return None
    return int(round((z ** 2) * p * (1 - p) / (halfwidth ** 2)))


def mde_paired_rate(n, discordance=0.25, power=0.80):
    """Minimum detectable difference between two paired rates.

    Paired binary comparisons are governed by the *discordant* pairs -- items where
    the two conditions disagree -- not by n alone. `discordance` is the expected
    share of such pairs; 0.25 is a moderate default and should be replaced with the
    observed value once the pilot has run.
    """
    if n <= 0 or discordance <= 0:
        return None
    z_beta = _Z.get(power, _Z[0.80])
    # McNemar normal approximation: detectable difference in proportions.
    return (_Z_ALPHA_TWO_SIDED + z_beta) * (discordance / n) ** 0.5


def paired_diff_by_qid(rows_a, rows_b, value_fn, drop_invalid=True):
    """Paired bootstrap of `value_fn`, aligned on qid across two conditions.

    Handles the two things that quietly break naive pairing: repeated samples per
    question (collapsed to a per-question mean first) and INVALID rows (dropped
    from *both* sides so the pairing stays intact).
    """
    def by_qid(rows):
        out = {}
        for r in rows:
            if drop_invalid and r.get("outcome") == "INVALID":
                continue
            out.setdefault(r.get("qid"), []).append(float(value_fn(r) or 0.0))
        return {q: sum(v) / len(v) for q, v in out.items()}

    a_map, b_map = by_qid(rows_a), by_qid(rows_b)
    shared = [q for q in a_map if q in b_map]
    if not shared:
        return None
    return paired_bootstrap_diff([a_map[q] for q in shared], [b_map[q] for q in shared])
