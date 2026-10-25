"""Build the evaluation set from the SQuAD v2.0 dev split.

Design
------
* Articles are shuffled with a fixed seed and split in two:
  - **eval articles**: every paragraph becomes a document in the base corpus, and
    questions are sampled only from these articles. Paragraphs from the same
    article are on-topic hard negatives for each other.
  - **distractor articles**: their paragraphs are the noise pool. Noise level n
    adds the first n paragraphs of a fixed shuffled order, so noise sets are
    nested and comparisons across levels are paired.
* Answerable and unanswerable questions are sampled separately to fixed counts.
  SQuAD v2 unanswerable questions were written against a real paragraph and
  look answerable from it, so retrieval usually returns that paragraph: the
  correct behaviour is to abstain.
* Each answerable question also gets a **counterfactual** copy of its source
  paragraph in which the gold answer span is replaced by a plausible answer of
  the same coarse type taken from another question. Experiment E2b puts these
  copies in the index to test whether the model follows conflicting evidence.
"""
import json
import os
import random
import re
from collections import defaultdict

import config


def _coarse_type(ans: str) -> str:
    a = ans.strip()
    if re.fullmatch(r"(1[0-9]|20)\d\d", a):
        return "year"
    if re.search(r"\d", a):
        return "number"
    if a[:1].isupper():
        return "proper"
    return "other"


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def build_dataset(seed: int = config.SEED,
                  n_eval_articles: int = config.N_EVAL_ARTICLES,
                  n_answerable: int = config.N_ANSWERABLE,
                  n_unanswerable: int = config.N_UNANSWERABLE,
                  raw_path: str = None,
                  out_path: str = config.DATASET_PATH) -> dict:
    raw_path = raw_path or os.path.join(config.BASE_DIR, config.SQUAD_PATH_RAW)
    with open(raw_path, encoding="utf-8") as f:
        squad = json.load(f)
    if squad.get("version") != "v2.0":
        raise ValueError(f"Expected SQuAD v2.0, got {squad.get('version')!r}")

    rng = random.Random(seed)
    articles = list(squad["data"])
    rng.shuffle(articles)
    eval_articles = articles[:n_eval_articles]
    noise_articles = articles[n_eval_articles:]

    docs, answerable, unanswerable = [], [], []
    for ai, art in enumerate(eval_articles):
        for pi, para in enumerate(art["paragraphs"]):
            doc_id = f"e{ai:02d}p{pi:03d}"
            docs.append({"doc_id": doc_id, "title": art["title"], "text": para["context"]})
            for qa in para["qas"]:
                item = {
                    "qid": qa["id"],
                    "question": qa["question"],
                    "doc_id": doc_id,
                    "title": art["title"],
                    "is_impossible": bool(qa.get("is_impossible")),
                }
                if item["is_impossible"]:
                    item["answers"] = []
                    unanswerable.append(item)
                else:
                    seen, answers, starts = set(), [], []
                    for a in qa["answers"]:
                        if _norm(a["text"]) and _norm(a["text"]) not in seen:
                            seen.add(_norm(a["text"]))
                            answers.append(a["text"])
                            starts.append(a["answer_start"])
                    item["answers"] = answers
                    item["answer_start"] = starts[0]
                    answerable.append(item)

    if len(answerable) < n_answerable or len(unanswerable) < n_unanswerable:
        raise ValueError("Not enough questions in the eval articles; raise N_EVAL_ARTICLES.")

    answerable = rng.sample(answerable, n_answerable)
    unanswerable = rng.sample(unanswerable, n_unanswerable)

    distractors = []
    for ai, art in enumerate(noise_articles):
        for pi, para in enumerate(art["paragraphs"]):
            distractors.append({"doc_id": f"d{ai:02d}p{pi:03d}", "title": art["title"],
                                "text": para["context"]})
    rng.shuffle(distractors)

    # --- counterfactual paragraphs for answerable questions
    doc_text = {d["doc_id"]: d["text"] for d in docs}
    pool = defaultdict(list)
    for q in answerable:
        pool[_coarse_type(q["answers"][0])].append(q["answers"][0])
    counterfactuals = []
    for q in answerable:
        gold = q["answers"][0]
        text = doc_text[q["doc_id"]]
        start = q["answer_start"]
        if text[start:start + len(gold)] != gold:
            start = text.find(gold)
        candidates = [a for a in pool[_coarse_type(gold)]
                      if _norm(a) not in {_norm(g) for g in q["answers"]}
                      and _norm(a) not in _norm(text)]
        if start < 0 or not candidates:
            q["cf_answer"] = None
            continue
        sub = rng.choice(candidates)
        q["cf_answer"] = sub
        counterfactuals.append({"doc_id": f"cf_{q['qid']}", "title": q["title"],
                                "text": text[:start] + sub + text[start + len(gold):],
                                "for_qid": q["qid"]})

    questions = answerable + unanswerable
    rng.shuffle(questions)
    for q in questions:
        q.pop("answer_start", None)

    dataset = {
        "meta": {
            "source": "SQuAD v2.0 dev",
            "seed": seed,
            "eval_articles": [a["title"] for a in eval_articles],
            "n_docs": len(docs),
            "n_distractors": len(distractors),
            "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable,
            "n_counterfactuals": len(counterfactuals),
        },
        "docs": docs,
        "distractors": distractors,
        "counterfactuals": counterfactuals,
        "questions": questions,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    return dataset


def load_dataset(path: str = config.DATASET_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python run.py build-data")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
