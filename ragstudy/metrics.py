"""Scoring and statistics.

Answer scoring follows the official SQuAD v2 evaluation script: answers are
normalised (lower-case, punctuation and articles removed) and EM/F1 take the max
over all gold answers. Abstention is detected from the UNANSWERABLE token plus
common refusal phrasings, which matters for the "open" prompt where the model
was never told the token.
"""
import collections
import re
import string

import numpy as np

import config


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds) -> float:
    return float(max((normalize_answer(pred) == normalize_answer(g) for g in golds), default=False))


def _f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1(pred: str, golds) -> float:
    return max((_f1(pred, g) for g in golds), default=0.0)


_REFUSAL = re.compile(
    r"unanswerable|cannot (?:be )?answer|can ?not (?:be )?determined|can't answer|"
    r"(?:does|do) not (?:contain|mention|provide|state|specify|say)|"
    r"doesn't (?:contain|mention|provide|state|specify|say)|not (?:contained|mentioned|provided|"
    r"stated|specified|given|found) in|no (?:information|mention)|i (?:do not|don't) know|unknown",
    re.I)


def is_abstention(pred: str) -> bool:
    return not (pred or "").strip() or bool(_REFUSAL.search(pred))


def score_record(rec: dict) -> dict:
    """Adds metric fields to one prediction record in place and returns it."""
    pred = rec["prediction"]
    abst = is_abstention(pred)
    rec["abstained"] = abst
    if rec["is_impossible"]:
        rec["correct_abstain"] = abst
        rec["hallucinated"] = not abst
        rec["v2_em"] = float(abst)
    else:
        rec["em"] = 0.0 if abst else exact_match(pred, rec["answers"])
        rec["f1"] = 0.0 if abst else f1(pred, rec["answers"])
        rec["over_refusal"] = abst
        rec["v2_em"] = rec["em"]
        if rec.get("cf_answer"):
            rec["misled"] = (not abst) and exact_match(pred, [rec["cf_answer"]]) == 1.0
    return rec


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(values, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED, alpha=0.05):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "n": int(v.size)}


def paired_bootstrap(a, b, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED, alpha=0.05):
    """Difference mean(b) - mean(a) over the same items, with CI and two-sided p."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired_bootstrap needs equal-length, non-empty arrays")
    d = b - a
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    obs = d.mean()
    centred = means - obs
    p = float(np.mean(np.abs(centred) >= abs(obs)))
    return {"diff": float(obs), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "p": p, "n": int(d.size)}


def summarise(records) -> dict:
    ans = [r for r in records if not r["is_impossible"]]
    una = [r for r in records if r["is_impossible"]]
    out = {"n": len(records), "n_answerable": len(ans), "n_unanswerable": len(una)}

    def put(name, rows, field):
        vals = [float(r[field]) for r in rows if r.get(field) is not None]
        if vals:
            out[name] = bootstrap_ci(vals)

    put("em", ans, "em")
    put("f1", ans, "f1")
    put("over_refusal", ans, "over_refusal")
    put("hallucination_rate", una, "hallucinated")
    put("abstention_rate", una, "correct_abstain")
    put("v2_em", records, "v2_em")
    put("retrieval_hit", records, "retrieval_hit")
    put("mrr", records, "rr")
    put("faithfulness", [r for r in ans if not r["abstained"]], "faithful")
    put("misled_rate", [r for r in ans if "misled" in r], "misled")
    # Of the unanswerable questions the model answered anyway, how many answers the
    # judge still found "supported" by the context (i.e. lifted from a near-miss passage).
    put("hallucination_grounded", [r for r in una if r.get("hallucinated")], "faithful")
    return out
