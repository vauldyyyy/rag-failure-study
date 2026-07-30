"""Build the SQuAD v2 retrieval corpus and the stratified evaluation set.

Two things this file must get right, because every downstream number depends on them:

1. The corpus is built from *every* paragraph in the dev set (1,204 paragraphs),
   not just the paragraphs the sampled questions came from. Retrieval over a
   corpus that only contains gold documents is not retrieval; it inflates P@K and
   makes the noise study meaningless.

2. Unanswerable questions (`is_impossible`) are kept. They are the only clean
   probe for hallucination-under-abstention-pressure: the gold paragraph is in
   the index, it is topically perfect, and it does not contain an answer.
"""

import json
import os
import random
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.chunker import chunk_text

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
TEMP_JSON = os.path.join(config.DATA_DIR, "squad_dev.json")


def load_and_process_squad(n_qa=config.N_QA,
                           frac_unanswerable=config.FRAC_UNANSWERABLE,
                           seed=config.SEED):
    if not os.path.exists(TEMP_JSON):
        print("Downloading SQuAD v2 dev set...")
        urllib.request.urlretrieve(SQUAD_URL, TEMP_JSON)

    with open(TEMP_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = []
    answerable, unanswerable = [], []

    print("Chunking every dev paragraph into the retrieval corpus...")
    doc_idx = 0
    for article in data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            doc_id = f"p{doc_idx:05d}"
            doc_idx += 1

            for chunk_i, text in enumerate(
                chunk_text(paragraph["context"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            ):
                corpus.append({
                    "chunk_id": f"{doc_id}_c{chunk_i}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                })

            for qa in paragraph["qas"]:
                # Dev answers are multi-annotator; SQuAD scoring takes the max over
                # all of them, so keep the de-duplicated set rather than answers[0].
                golds = list(dict.fromkeys(a["text"] for a in qa.get("answers", [])))
                record = {
                    "qid": qa["id"],
                    "question": qa["question"],
                    "answers": golds,
                    "is_answerable": not qa.get("is_impossible", False),
                    "gold_doc_id": doc_id,
                    "title": title,
                }
                if record["is_answerable"] and golds:
                    answerable.append(record)
                elif not record["is_answerable"]:
                    unanswerable.append(record)

    rng = random.Random(seed)
    n_unans = int(round(n_qa * frac_unanswerable))
    n_ans = n_qa - n_unans

    if len(answerable) < n_ans or len(unanswerable) < n_unans:
        raise RuntimeError(
            f"Not enough questions to sample {n_ans}/{n_unans}: "
            f"pool has {len(answerable)} answerable, {len(unanswerable)} unanswerable."
        )

    qa_subset = rng.sample(answerable, n_ans) + rng.sample(unanswerable, n_unans)
    rng.shuffle(qa_subset)

    with open(config.SQUAD_SUBSET_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_subset, f, ensure_ascii=False, indent=2)
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    manifest = {
        "dataset": "squad_v2_dev",
        "seed": seed,
        "n_documents": doc_idx,
        "n_chunks": len(corpus),
        "n_qa": len(qa_subset),
        "n_answerable": n_ans,
        "n_unanswerable": n_unans,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "pool_answerable": len(answerable),
        "pool_unanswerable": len(unanswerable),
    }
    with open(os.path.join(config.DATA_DIR, "squad_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    load_and_process_squad()
