"""Context perturbations — the independent variables of the study.

All perturbations act on the *context handed to the generator*, not on the FAISS
index. That is a deliberate design change from index poisoning:

  * index poisoning makes the noise level a property of the whole run, so the
    amount of noise any single question actually receives is uncontrolled and
    unmeasured — you cannot say "this answer was produced under 50% noise";
  * context-level perturbation fixes the noise fraction exactly, per item, and
    holds retrieval identical across conditions, so the only thing that varies
    between two conditions is the thing named in the condition.

The four perturbations map one-to-one onto the failure categories the study
claims to isolate:

    RANDOM          irrelevant retrieval      (topically unrelated distractors)
    RELATED         partial retrieval         (adjacent, plausible, wrong)
    COUNTERFACTUAL  contradictory retrieval   (context asserts a false answer)
    DROP_GOLD       no answer in context      (evidence absent, question answerable)
"""

import random
import re

CLEAN = "clean"
RANDOM = "random"
RELATED = "related"
COUNTERFACTUAL = "counterfactual"
DROP_GOLD = "drop_gold"

ALL_NOISE_TYPES = [RANDOM, RELATED, COUNTERFACTUAL, DROP_GOLD]


def _gold_ids(qa):
    ids = qa.get("gold_doc_ids") or [qa.get("gold_doc_id")]
    return {g for g in ids if g}


def _split_gold(ranked, qa):
    gold = _gold_ids(qa)
    hits = [c for c in ranked if c["doc_id"] in gold]
    misses = [c for c in ranked if c["doc_id"] not in gold]
    return hits, misses


def apply_noise(ranked, qa, k, noise_type=CLEAN, noise_frac=0.0,
                corpus=None, rng=None, gold_position=None):
    """Build the k-slot context for one item.

    ranked        cached top-RETRIEVAL_DEPTH results for this query (never re-ranked)
    noise_frac    share of the k slots replaced by distractors, rounded to whole slots
    gold_position 0-based slot to move a retrieved gold chunk into, or None to leave
                  the ranking untouched (used by the position/lost-in-the-middle sweep)
    """
    rng = rng or random.Random(0)
    context = list(ranked[:k])

    if noise_type == CLEAN or noise_frac <= 0:
        pass

    elif noise_type == DROP_GOLD:
        gold = _gold_ids(qa)
        kept = [c for c in ranked if c["doc_id"] not in gold]
        context = kept[:k]

    elif noise_type == COUNTERFACTUAL:
        n_corrupt = max(1, int(round(k * noise_frac)))
        context = [dict(c) for c in context]
        corrupted = 0
        for c in context:
            if corrupted >= n_corrupt:
                break
            new_text = _corrupt(c["text"], qa, rng)
            if new_text != c["text"]:
                c["text"] = new_text
                c["perturbed"] = COUNTERFACTUAL
                corrupted += 1

    else:
        n_noise = int(round(k * noise_frac))
        if n_noise <= 0:
            return context

        if noise_type == RANDOM:
            if not corpus:
                raise ValueError("RANDOM noise needs the corpus to sample distractors from")
            gold = _gold_ids(qa)
            pool = [c for c in corpus if c["doc_id"] not in gold]
            distractors = [
                {"score": 0.0, "text": d["text"], "doc_id": d["doc_id"],
                 "chunk_id": d.get("chunk_id", ""), "title": d.get("title", ""),
                 "perturbed": RANDOM}
                for d in rng.sample(pool, min(n_noise, len(pool)))
            ]
        else:  # RELATED — real retrieval results from deeper in the ranking
            _, misses = _split_gold(ranked, qa)
            tail = [c for c in misses if c not in context] or misses
            distractors = [dict(c, perturbed=RELATED) for c in tail[:n_noise]]

        # Replace the lowest-ranked slots first so a gold chunk at rank 1 survives
        # until the noise fraction is high enough to displace it.
        keep = k - len(distractors)
        context = context[:max(keep, 0)] + distractors

    if gold_position is not None and context:
        gold = _gold_ids(qa)
        idx = next((i for i, c in enumerate(context) if c["doc_id"] in gold), None)
        if idx is not None:
            pos = min(gold_position, len(context) - 1)
            chunk = context.pop(idx)
            context.insert(pos, chunk)

    return context


_NUM_RE = re.compile(r"\b\d{1,4}\b")


def _corrupt(text, qa, rng):
    """Make the context assert something false about this question.

    Prefers substituting the gold answer string, which produces a context that
    directly contradicts the reference; falls back to perturbing a number.
    """
    for gold in qa.get("answers") or []:
        if gold and gold in text:
            replacement = _alternative(gold, rng)
            if replacement != gold:
                return text.replace(gold, replacement)

    def bump(m):
        n = int(m.group())
        return str(n + rng.choice([-7, -3, 3, 7, 11]) if n > 12 else n + 1)

    return _NUM_RE.sub(bump, text, count=2)


_SUBSTITUTES = {
    "france": "Belgium", "germany": "Austria", "england": "Scotland",
    "london": "Manchester", "paris": "Lyon", "europe": "Asia",
}


def _alternative(gold, rng):
    key = gold.strip().lower()
    if key in _SUBSTITUTES:
        return _SUBSTITUTES[key]
    if gold.strip().isdigit():
        n = int(gold.strip())
        return str(n + rng.choice([-11, -5, 5, 11]))
    words = gold.split()
    if len(words) > 1:
        return " ".join(reversed(words))
    return gold[::-1].capitalize() if len(gold) > 3 else gold
