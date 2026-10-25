import numpy as np

from ragstudy import metrics


def test_normalisation_and_em():
    assert metrics.normalize_answer("The  Normans!") == "normans"
    assert metrics.exact_match("the Normans", ["Normans", "Norman people"]) == 1.0
    assert metrics.exact_match("Vikings", ["Normans"]) == 0.0


def test_f1_takes_best_gold():
    # vs "in 10th century": precision 2/2, recall 2/3 -> 0.8; vs "10th": 1/2, 1/1 -> 0.667
    assert abs(metrics.f1("10th century", ["in the 10th century", "10th"]) - 0.8) < 1e-9
    assert metrics.f1("", ["x"]) == 0.0


def test_abstention_detection():
    for s in ["UNANSWERABLE", "The context does not mention this.", "I don't know", ""]:
        assert metrics.is_abstention(s), s
    for s in ["Normandy", "1066", "Rollo"]:
        assert not metrics.is_abstention(s), s


def test_score_record_unanswerable():
    r = metrics.score_record({"prediction": "Rollo", "is_impossible": True, "answers": []})
    assert r["hallucinated"] and not r["correct_abstain"] and r["v2_em"] == 0.0
    r = metrics.score_record({"prediction": "UNANSWERABLE", "is_impossible": True, "answers": []})
    assert r["correct_abstain"] and r["v2_em"] == 1.0


def test_score_record_misled():
    r = metrics.score_record({"prediction": "1215", "is_impossible": False,
                              "answers": ["1066"], "cf_answer": "1215"})
    assert r["misled"] and r["em"] == 0.0


def test_bootstrap_ci_brackets_mean():
    ci = metrics.bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1])
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_paired_bootstrap_detects_large_shift():
    a = np.zeros(200)
    b = np.r_[np.ones(150), np.zeros(50)]
    t = metrics.paired_bootstrap(a, b)
    assert t["diff"] == 0.75 and t["p"] < 0.01 and t["lo"] > 0
