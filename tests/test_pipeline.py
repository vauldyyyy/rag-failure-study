"""End-to-end check with the offline mock backend (no API key needed)."""
import json
import os

import pytest

import config
from ragstudy import data, study
from ragstudy.backends import MockBackend
from ragstudy.cache import Cache


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "dataset.json"
    return data.build_dataset(n_answerable=40, n_unanswerable=20, out_path=str(out))


def test_dataset_is_disjoint_and_balanced(dataset):
    qs = dataset["questions"]
    assert sum(q["is_impossible"] for q in qs) == 20
    doc_ids = {d["doc_id"] for d in dataset["docs"]}
    assert all(q["doc_id"] in doc_ids for q in qs)
    eval_titles = set(dataset["meta"]["eval_articles"])
    assert not eval_titles & {d["title"] for d in dataset["distractors"]}
    assert len(dataset["distractors"]) >= max(config.NOISE_LEVELS)


def test_counterfactuals_replace_the_answer(dataset):
    docs = {d["doc_id"]: d["text"] for d in dataset["docs"]}
    qs = {q["qid"]: q for q in dataset["questions"]}
    assert dataset["counterfactuals"]
    for cf in dataset["counterfactuals"]:
        q = qs[cf["for_qid"]]
        assert q["cf_answer"] in cf["text"]
        assert cf["text"] != docs[q["doc_id"]]


def test_all_experiments_run_offline(dataset, tmp_path):
    mock = MockBackend()
    st = study.Study(dataset, mock, mock, cache=Cache(":memory:"), judge=True,
                     results_dir=str(tmp_path), log=lambda *_: None)
    for exp, conds in study.experiments().items():
        summ = st.run(exp)
        assert set(summ["conditions"]) == {c["name"] for c in conds}
        path = os.path.join(st.out_root, exp, "summary.json")
        assert json.load(open(path))["experiment"] == exp
    # the gold passage sits exactly where E4 put it
    rec = json.loads(open(os.path.join(st.out_root, "E4", "pos_5.jsonl")).readline())
    q = next(q for q in dataset["questions"] if q["qid"] == rec["qid"])
    assert rec["context_ids"][4] == q["doc_id"] and len(rec["context_ids"]) == config.POSITION_K
    # more distractors never make the index smaller
    assert st.index(500, False)[0].ntotal == len(dataset["docs"]) + 500


def test_report_builds(dataset, tmp_path):
    from analysis.report import build_report
    mock = MockBackend()
    st = study.Study(dataset, mock, mock, cache=Cache(":memory:"), results_dir=str(tmp_path),
                     log=lambda *_: None)
    for exp in ("E1", "E2", "E4", "E5"):
        st.run(exp)
    msg = build_report(results_dir=str(tmp_path))
    assert "RESULTS.md" in msg
    assert os.path.exists(os.path.join(st.out_root, "figures", "e2_noise.png"))
