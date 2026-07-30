"""Second evaluation domain: HotpotQA (distractor setting).

Why a second dataset: a failure curve measured on one benchmark is a property of
that benchmark until it replicates. HotpotQA adds two things SQuAD cannot give:

  * multi-hop questions, where partial retrieval is the *normal* case (one of two
    gold paragraphs retrieved) rather than an injected condition;
  * human-selected hard distractors — 8 topically adjacent wrong paragraphs per
    question — which are exactly the "semantically adjacent" noise the study cares
    about, without us having to synthesise them.

HotpotQA has no unanswerable split, so the abstention analysis stays SQuAD-only;
this file exists to test whether the noise/partial-retrieval findings generalise.

Output schema is identical to load_squad.py so every experiment runs unchanged.
"""

import json
import os
import random
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.chunker import chunk_text

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
TEMP_JSON = os.path.join(config.DATA_DIR, "hotpot_dev_distractor.json")


def load_and_process_hotpot(n_qa=config.N_QA, seed=config.SEED):
    if not os.path.exists(TEMP_JSON):
        print("Downloading HotpotQA dev (distractor) — ~45 MB...")
        urllib.request.urlretrieve(HOTPOT_URL, TEMP_JSON)

    with open(TEMP_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    sample = rng.sample(data, min(n_qa, len(data)))

    corpus = []
    seen_titles = {}
    qa_subset = []

    for item in sample:
        gold_titles = {t for t, _ in item["supporting_facts"]}
        gold_doc_ids = []

        for title, sentences in item["context"]:
            if title not in seen_titles:
                doc_id = f"h{len(seen_titles):05d}"
                seen_titles[title] = doc_id
                text = " ".join(sentences)
                for chunk_i, chunk in enumerate(
                    chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                ):
                    corpus.append({
                        "chunk_id": f"{doc_id}_c{chunk_i}",
                        "doc_id": doc_id,
                        "title": title,
                        "text": chunk,
                    })
            if title in gold_titles:
                gold_doc_ids.append(seen_titles[title])

        qa_subset.append({
            "qid": item["_id"],
            "question": item["question"],
            "answers": [item["answer"]],
            "is_answerable": True,
            # Primary gold doc keeps the single-id contract used by P@K; the full
            # set is kept separately so multi-hop coverage can be scored properly.
            "gold_doc_id": gold_doc_ids[0] if gold_doc_ids else None,
            "gold_doc_ids": gold_doc_ids,
            "title": item.get("type", ""),
        })

    with open(config.HOTPOT_SUBSET_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_subset, f, ensure_ascii=False, indent=2)
    with open(config.HOTPOT_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    manifest = {
        "dataset": "hotpotqa_dev_distractor",
        "seed": seed,
        "n_documents": len(seen_titles),
        "n_chunks": len(corpus),
        "n_qa": len(qa_subset),
        "n_answerable": len(qa_subset),
        "n_unanswerable": 0,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }
    with open(os.path.join(config.DATA_DIR, "hotpot_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    load_and_process_hotpot()
