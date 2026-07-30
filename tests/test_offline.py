"""Offline test suite — everything that does not need the Gemini API.

Run with:  python tests/test_offline.py

Covers the parts where a silent bug would corrupt every published number:
answer scoring, abstention detection, the noise perturbations, and the statistics.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (normalize_answer, exact_match_score, f1_score,
                           is_abstention, score_item, aggregate, recall_at_k)
from utils import noise as nz
from utils.stats import bootstrap_ci, paired_bootstrap_diff, auroc
from utils.rerank import bm25_rerank
from utils.chunker import chunk_text

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    ok &= bool(cond)

# --- lexical ---
check("EM max-over-golds", exact_match_score("France", ["the France", "Belgium"]) == 1)
check("F1 multiset not set",
      abs(f1_score("France France France", ["France"]) - 0.5) < 1e-6,
      f"got {f1_score('France France France', ['France']):.4f}")
# 4 predicted tokens, 1 gold token -> P=1/4, R=1 -> F1=0.4
check("F1 partial", abs(f1_score("in the year 1241 AD", ["1241"]) - 0.4) < 1e-6,
      f"{f1_score('in the year 1241 AD', ['1241']):.3f}")
check("empty pred", f1_score("", ["x"]) == 0)

# --- abstention ---
check("plain refusal", is_abstention("I cannot answer based on the context."))
check("short refusal variant", is_abstention("The context does not contain this information."))
check("hedged assertion is NOT abstention",
      not is_abstention("The context does not explicitly say, but the Mongols invaded "
                        "Hungary in 1241 and then moved on to Poland shortly after, "
                        "so the answer is most likely 1241 based on the surrounding text."))
check("normal answer", not is_abstention("Normandy is located in France."))

# --- noise ---
ranked = [{"score": 0.9 - i*0.05, "text": f"chunk {i} about topic", "doc_id": f"p{i}",
           "chunk_id": f"p{i}_c0", "title": "T"} for i in range(20)]
qa = {"qid": "q1", "question": "where?", "answers": ["France"], "is_answerable": True,
      "gold_doc_id": "p0"}
corpus = [{"chunk_id": f"x{i}", "doc_id": f"x{i}", "text": f"unrelated {i}", "title": "X"}
          for i in range(50)]
rng = random.Random(0)

c = nz.apply_noise(ranked, qa, 4, nz.CLEAN, 0.0, corpus, rng)
check("clean = top-k", [x["doc_id"] for x in c] == ["p0", "p1", "p2", "p3"])

c = nz.apply_noise(ranked, qa, 4, nz.RELATED, 0.5, corpus, rng)
check("related@50 keeps 2 + 2 distractors", len(c) == 4 and sum(1 for x in c if x.get("perturbed")) == 2,
      [x["doc_id"] for x in c])
check("related@50 keeps gold at rank1", c[0]["doc_id"] == "p0")

c = nz.apply_noise(ranked, qa, 4, nz.RELATED, 1.0, corpus, rng)
check("related@100 drops gold", all(x["doc_id"] != "p0" for x in c), [x["doc_id"] for x in c])

c = nz.apply_noise(ranked, qa, 4, nz.RANDOM, 0.5, corpus, rng)
check("random@50 pulls from corpus", sum(1 for x in c if x["doc_id"].startswith("x")) == 2)

c = nz.apply_noise(ranked, qa, 4, nz.DROP_GOLD, 1.0, corpus, rng)
check("drop_gold removes gold doc", all(x["doc_id"] != "p0" for x in c) and len(c) == 4)

gold_qa = dict(qa)
gr = [{"score": 0.9, "text": "Normandy is in France today.", "doc_id": "p0",
       "chunk_id": "a", "title": "T"}] + ranked[1:]
c = nz.apply_noise(gr, gold_qa, 4, nz.COUNTERFACTUAL, 0.5, corpus, rng)
check("counterfactual rewrites gold text", "France" not in c[0]["text"], c[0]["text"])

c = nz.apply_noise(ranked, qa, 4, nz.CLEAN, 0.0, corpus, rng, gold_position=2)
check("gold moved to slot 2", c[2]["doc_id"] == "p0", [x["doc_id"] for x in c])

# --- scoring / aggregate ---
ctx = [{"text": "Normandy is in France.", "doc_id": "p0"}]
r1 = score_item(qa, "France", "ok", ctx, use_judge=False)
check("correct scored", r1["outcome"] == "CORRECT" and r1["gold_in_context"], r1["outcome"])

unans = {"qid": "q2", "question": "who walked on Mars?", "answers": [],
         "is_answerable": False, "gold_doc_id": "p0"}
r2 = score_item(unans, "Neil Armstrong in 1969.", "ok", ctx, use_judge=False)
check("unanswerable asserted = UNSUPPORTED (no judge call)", r2["outcome"] == "UNSUPPORTED")
r3 = score_item(unans, "I cannot answer based on the context.", "ok", ctx, use_judge=False)
check("unanswerable refused = ABSTAIN", r3["outcome"] == "ABSTAIN")
r4 = score_item(qa, "", "blocked", ctx, use_judge=False)
check("blocked = INVALID", r4["outcome"] == "INVALID")

r5 = score_item(qa, "France", "ok", [{"text": "unrelated", "doc_id": "zz"}], use_judge=False)
check("parametric flag when gold absent but correct", r5.get("parametric") is True)

agg = aggregate([r1, r2, r3, r4, r5])
check("aggregate excludes INVALID", agg["n"] == 4 and agg["n_invalid"] == 1, agg)
check("split halluc rates", agg["hallucination_rate_unanswerable"] == 0.5, agg["hallucination_rate_unanswerable"])
check("silent failure rate present", agg["silent_failure_rate"] is not None, agg["silent_failure_rate"])

check("recall_at_k multi-hop", recall_at_k(["a", "b"], ["a", "c"]) == 0.5)

# --- stats ---
ci = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
check("bootstrap ci brackets mean", ci["lo"] <= ci["mean"] <= ci["hi"], ci)
d = paired_bootstrap_diff([0]*50 + [1]*50, [1]*100)
check("paired diff detects shift", d["diff"] == 0.5 and d["p_two_sided"] < 0.05, d)
check("auroc perfect", auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0)
check("auroc chance", auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5)
check("auroc single class", auroc([1, 2], [1, 1]) is None)

# --- rerank ---
ctxs = [{"text": "completely unrelated filler text", "doc_id": "a"},
        {"text": "Normandy is a region in France with a long history", "doc_id": "b"}]
rr = bm25_rerank("Where is Normandy located in France?", ctxs, keep=1)
check("bm25 surfaces relevant chunk", rr[0]["doc_id"] == "b", rr[0]["doc_id"])

# --- chunker ---
long_text = " ".join(["word"] * 900)
ch = chunk_text(long_text, 256, 50)
check("chunker splits long text", len(ch) >= 3, len(ch))
check("chunker no empty tail", all(c.strip() for c in ch))
check("chunker short text = 1 chunk", len(chunk_text("hello world", 256, 50)) == 1)

# --- repeated-sampling safety -------------------------------------------------
from utils.stats import collapse_repeats, paired_diff_by_qid

rep_rows = ([{"qid": "a", "outcome": "CORRECT", "f1": 1.0}] * 2
            + [{"qid": "a", "outcome": "ABSTAIN", "f1": 0.0}] * 2
            + [{"qid": "b", "outcome": "CORRECT", "f1": 1.0}] * 4)
collapsed = collapse_repeats(rep_rows, lambda r: r["outcome"] == "CORRECT")
check("collapse_repeats yields one value per question", len(collapsed) == 2, collapsed)
check("collapse_repeats averages within question", sorted(collapsed) == [0.5, 1.0], collapsed)

# 4 repeats of 100 items must not shrink the interval the way 400 rows would.
many = [{"qid": f"q{i}", "outcome": "CORRECT" if i % 2 else "ABSTAIN", "f1": 0.0}
        for i in range(100)]
ci_single = bootstrap_ci(collapse_repeats(many, lambda r: r["outcome"] == "CORRECT"))
ci_repeat = bootstrap_ci(collapse_repeats(many * 4, lambda r: r["outcome"] == "CORRECT"))
check("repeats do not inflate n", ci_single["n"] == ci_repeat["n"] == 100,
      f"{ci_single['n']} vs {ci_repeat['n']}")

a_rows = [{"qid": f"q{i}", "outcome": "ABSTAIN", "f1": 0.0} for i in range(60)]
b_rows = [{"qid": f"q{i}", "outcome": "CORRECT", "f1": 1.0} for i in range(60)]
b_rows.append({"qid": "q0", "outcome": "INVALID", "f1": None})
d = paired_diff_by_qid(a_rows, b_rows, lambda r: r["f1"])
check("paired_diff_by_qid aligns and drops INVALID", d["n"] == 60 and d["diff"] == 1.0, d)
check("paired_diff_by_qid handles disjoint qids",
      paired_diff_by_qid(a_rows, [{"qid": "zz", "outcome": "CORRECT", "f1": 1.0}],
                         lambda r: r["f1"]) is None)

# --- dry-run stub -------------------------------------------------------------
from utils.stub_llm import stub_generate, stub_judge

g1 = stub_generate("where?", [{"doc_id": "p0", "text": "Normandy is in France."}],
                   gold_answers=["France"], gold_doc_ids=["p0"], is_answerable=True)
g2 = stub_generate("where?", [{"doc_id": "p0", "text": "Normandy is in France."}],
                   gold_answers=["France"], gold_doc_ids=["p0"], is_answerable=True)
check("stub generator is deterministic", g1 == g2, g1)
check("stub returns llm-shaped dict", set(g1) == {"text", "status"} and g1["status"] == "ok")
check("stub judge detects refusal",
      stub_judge("q", [], "I cannot answer based on the context.") == "REFUSAL")
check("stub judge grounds in context",
      stub_judge("q", [{"text": "Normandy is in France"}], "Normandy France") == "SUPPORTED")
check("stub judge flags ungrounded",
      stub_judge("q", [{"text": "Normandy is in France"}], "Berlin Germany Prussia") == "UNSUPPORTED")

# --- fatal-error classification ----------------------------------------------
from utils.llm import is_fatal_error

check("auth errors are fatal", is_fatal_error(Exception("API key not valid")))
check("bad model is fatal", is_fatal_error(Exception("models/foo is not found")))
check("rate limits are retryable", not is_fatal_error(Exception("429 Resource exhausted")))

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
