"""Experiment definitions and the runner.

Experiments (all share one dataset, one index and one cache):

E1  baseline        K=3, no noise, default prompt.
E2  noise           K=3, default prompt, {0,100,250,500} off-topic distractor paragraphs.
E2b conflict        K=3, default prompt, counterfactual copies of gold paragraphs in the
                    index (answerable questions only). Measures how often the model
                    repeats the planted wrong answer when it is retrieved.
E3  context size    no noise, default prompt, K in {1,2,3,5,7,10}.
E4  position        answerable only; 10 passages with the gold paragraph forced to
                    position {1,3,5,7,10} among the 9 best-scoring non-gold passages
                    ("lost in the middle" test).
E5  prompting       K=3, prompts {open, default, strict} x noise {0, max}.

Identical prompts recur across experiments (E1 = E2/noise 0 = E3/K 3 = E5/default,
noise 0); the generation cache makes repeats free.
"""
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import faiss
import numpy as np

import config
from . import metrics, prompts
from .cache import Cache, key_of


def experiments():
    max_noise = max(config.NOISE_LEVELS)
    base = dict(k=config.DEFAULT_K, noise=0, prompt="default", position=None, cf=False,
                subset="all")
    return {
        "E1": [dict(base, name="baseline")],
        "E2": [dict(base, name=f"noise_{n}", noise=n) for n in config.NOISE_LEVELS],
        "E2b": [dict(base, name="conflict", cf=True, subset="answerable_cf")],
        "E3": [dict(base, name=f"k_{k}", k=k) for k in config.K_VALUES],
        "E4": [dict(base, name=f"pos_{p}", k=config.POSITION_K, position=p, subset="answerable")
               for p in config.POSITIONS],
        "E5": [dict(base, name=f"{pr}_noise_{n}", prompt=pr, noise=n)
               for pr in config.PROMPTS for n in sorted({0, max_noise})],
    }


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.BASE_DIR,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


class Study:
    def __init__(self, dataset, gen, emb, cache: Cache = None, judge=False, limit=None,
                 workers=1, results_dir=config.RESULTS_DIR, log=print):
        self.ds, self.gen, self.emb = dataset, gen, emb
        self.cache = cache or Cache(config.CACHE_PATH)
        self.judge, self.workers, self.log = judge, workers, log
        qs = dataset["questions"]
        self.questions = qs[:limit] if limit else qs
        self.out_root = os.path.join(results_dir, _slug(gen.name))
        self.docs = {d["doc_id"]: d for d in
                     dataset["docs"] + dataset["distractors"] + dataset["counterfactuals"]}
        self._vecs = None
        self._indices = {}

    # ------------------------------------------------------------------ embeddings
    def _embed(self, texts, task):
        keys = [key_of(self.emb.embed_name, task, t) for t in texts]
        out = [self.cache.get_emb(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            self.log(f"  embedding {len(todo)} texts ({task}) ...")
            for s in range(0, len(todo), 500):
                chunk = todo[s:s + 500]
                vecs = self.emb.embed([texts[i] for i in chunk], task)
                self.cache.put_embs([(keys[i], v) for i, v in zip(chunk, vecs)])
                for i, v in zip(chunk, vecs):
                    out[i] = np.asarray(v, dtype=np.float32)
        m = np.vstack(out).astype(np.float32)
        faiss.normalize_L2(m)
        return m

    def vectors(self):
        if self._vecs is None:
            ids = list(self.docs)
            dm = self._embed([self.docs[i]["text"] for i in ids], "RETRIEVAL_DOCUMENT")
            qm = self._embed([q["question"] for q in self.questions], "RETRIEVAL_QUERY")
            self._vecs = ({i: n for n, i in enumerate(ids)}, dm, qm)
        return self._vecs

    def index(self, noise: int, cf: bool):
        key = (noise, cf)
        if key not in self._indices:
            row_of, dm, _ = self.vectors()
            ids = [d["doc_id"] for d in self.ds["docs"]]
            ids += [d["doc_id"] for d in self.ds["distractors"][:noise]]
            if cf:
                ids += [d["doc_id"] for d in self.ds["counterfactuals"]]
            idx = faiss.IndexFlatIP(dm.shape[1])
            idx.add(dm[[row_of[i] for i in ids]])
            self._indices[key] = (idx, ids)
        return self._indices[key]

    def retrieve(self, qi: int, noise: int, cf: bool, k: int):
        idx, ids = self.index(noise, cf)
        _, _, qm = self.vectors()
        scores, rows = idx.search(qm[qi:qi + 1], k)
        return [(ids[r], float(s)) for r, s in zip(rows[0], scores[0]) if r != -1]

    # ------------------------------------------------------------------ one question
    def _generate(self, system, prompt, backend_model=None):
        model = backend_model or self.gen.gen_model
        k = key_of("gen", self.gen.name, model, system, prompt, config.MAX_OUTPUT_TOKENS,
                   getattr(self.gen, "thinking_budget", None))
        hit = self.cache.get_gen(k)
        if hit is None:
            hit = self.gen.generate(system, prompt, model=model)
            self.cache.put_gen(k, hit)
        return hit

    def _contexts(self, qi, q, cond):
        if cond["position"] is not None:
            ranked = self.retrieve(qi, cond["noise"], cond["cf"], cond["k"] + 1)
            others = [d for d, _ in ranked if d != q["doc_id"]][:cond["k"] - 1]
            ids = others[:cond["position"] - 1] + [q["doc_id"]] + others[cond["position"] - 1:]
            return ids, None
        ranked = self.retrieve(qi, cond["noise"], cond["cf"], max(cond["k"], 10))
        ids = [d for d, _ in ranked]
        rank = ids.index(q["doc_id"]) + 1 if q["doc_id"] in ids else None
        return ids[:cond["k"]], rank

    def run_one(self, qi, cond):
        q = self.questions[qi]
        ids, gold_rank = self._contexts(qi, q, cond)
        passages = [self.docs[i] for i in ids]
        system = prompts.SYSTEM[cond["prompt"]]
        res = self._generate(system, prompts.answer_prompt(q["question"], passages))
        rec = {
            "qid": q["qid"], "question": q["question"], "is_impossible": q["is_impossible"],
            "answers": q["answers"], "context_ids": ids, "prediction": res["text"],
            "finish_reason": res.get("finish_reason"),
        }
        if cond["position"] is None:
            rec["retrieval_hit"] = float(q["doc_id"] in ids)
            rec["rr"] = 1.0 / gold_rank if gold_rank and gold_rank <= 10 else 0.0
        cf_id = f"cf_{q['qid']}"
        if cond["cf"] and cf_id in ids:
            rec["cf_answer"] = q.get("cf_answer")
            rec["gold_in_context"] = q["doc_id"] in ids
        metrics.score_record(rec)
        if self.judge and not rec["abstained"]:
            j = self._generate(prompts.JUDGE_SYSTEM,
                               prompts.judge_prompt(q["question"], passages, rec["prediction"]),
                               backend_model=config.JUDGE_MODEL if self.gen.name.startswith("gemini")
                               else None)
            rec["faithful"] = j["text"].strip().upper().startswith("SUPPORTED")
        return rec

    # ------------------------------------------------------------------ conditions
    def subset(self, cond):
        qs = list(enumerate(self.questions))
        if cond["subset"] == "answerable":
            return [qi for qi, q in qs if not q["is_impossible"]]
        if cond["subset"] == "answerable_cf":
            return [qi for qi, q in qs if not q["is_impossible"] and q.get("cf_answer")]
        return [qi for qi, _ in qs]

    def run_condition(self, exp, cond):
        qis = self.subset(cond)
        self.log(f"[{exp}] {cond['name']}: {len(qis)} questions")
        done = [0]

        def work(qi):
            r = self.run_one(qi, cond)
            done[0] += 1
            if done[0] % 50 == 0:
                self.log(f"    {done[0]}/{len(qis)}")
            return r

        if self.workers > 1:
            with ThreadPoolExecutor(self.workers) as ex:
                recs = list(ex.map(work, qis))
        else:
            recs = [work(qi) for qi in qis]
        out_dir = os.path.join(self.out_root, exp)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{cond['name']}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return recs

    def run(self, exp):
        conds = experiments()[exp]
        summary = {"experiment": exp, "conditions": {}, "meta": self.meta()}
        for c in conds:
            recs = self.run_condition(exp, c)
            summary["conditions"][c["name"]] = {"settings": c, "metrics": metrics.summarise(recs)}
        with open(os.path.join(self.out_root, exp, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def meta(self):
        return {
            "generator": self.gen.name, "embedder": self.emb.embed_name,
            "judge": (config.JUDGE_MODEL if self.judge else None),
            "thinking_budget": getattr(self.gen, "thinking_budget", None),
            "n_questions": len(self.questions), "dataset": self.ds["meta"],
            "git_sha": _git_sha(), "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def plan(dataset, limit=None):
    """Upper bound on generation calls per experiment (before cache de-duplication)."""
    qs = dataset["questions"][:limit] if limit else dataset["questions"]
    n_all = len(qs)
    n_ans = sum(not q["is_impossible"] for q in qs)
    n_cf = sum(not q["is_impossible"] and bool(q.get("cf_answer")) for q in qs)
    size = {"all": n_all, "answerable": n_ans, "answerable_cf": n_cf}
    avg_doc_tokens = np.mean([len(d["text"]) for d in dataset["docs"]]) / 4
    rows, total_calls, total_tokens = [], 0, 0
    for exp, conds in experiments().items():
        calls = sum(size[c["subset"]] for c in conds)
        tokens = sum(size[c["subset"]] * (c["k"] * (avg_doc_tokens + 8) + 90) for c in conds)
        rows.append((exp, len(conds), calls, int(tokens)))
        total_calls += calls
        total_tokens += tokens
    return rows, total_calls, int(total_tokens)
