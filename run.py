"""Command-line entry point.

    python run.py build-data                     # sample questions, corpus, distractors
    python run.py plan                           # how many API calls a full run needs
    python run.py run E1 --limit 20              # quick check on 20 questions
    python run.py run all --judge                # full study with faithfulness judge
    python run.py run E5 --gen openai            # same experiment on another model
    python run.py report                         # tables, figures, results/RESULTS.md

Use --gen mock --embed mock to exercise the whole pipeline offline (numbers are
meaningless; the mock exists for testing).
"""
import argparse
import sys

import config
from ragstudy import data, study


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-data")
    pl = sub.add_parser("plan")
    pl.add_argument("--limit", type=int)

    r = sub.add_parser("run")
    r.add_argument("experiments", nargs="+", help="E1 E2 E2b E3 E4 E5, or 'all'")
    r.add_argument("--gen", choices=["gemini", "openai", "mock"], default="gemini")
    r.add_argument("--embed", choices=["gemini", "mock"], default=None,
                   help="defaults to gemini, or mock when --gen mock")
    r.add_argument("--limit", type=int, help="only the first N questions (for quick checks)")
    r.add_argument("--judge", action="store_true", help="run the LLM faithfulness judge")
    r.add_argument("--workers", type=int, default=1, help="parallel requests (paid tiers)")

    rep = sub.add_parser("report")
    rep.add_argument("--model", help="results subfolder (default: every model found)")

    a = p.parse_args(argv)

    if a.cmd == "build-data":
        ds = data.build_dataset()
        print(f"Built {config.DATASET_PATH}")
        for k, v in ds["meta"].items():
            print(f"  {k}: {v}")
        return 0

    if a.cmd == "plan":
        ds = data.load_dataset()
        rows, calls, tokens = study.plan(ds, a.limit)
        print(f"{'exp':5} {'conds':>5} {'gen calls':>10} {'~input tokens':>14}")
        for exp, n, c, t in rows:
            print(f"{exp:5} {n:5d} {c:10d} {t:14,d}")
        print(f"TOTAL {'':5} {calls:10d} {tokens:14,d}  (upper bound; repeated prompts are cached,"
              f" --judge adds up to one call per answered question)")
        return 0

    if a.cmd == "run":
        from ragstudy.backends import make_backends
        embed = a.embed or ("mock" if a.gen == "mock" else "gemini")
        gen, emb = make_backends(a.gen, embed)
        ds = data.load_dataset()
        st = study.Study(ds, gen, emb, judge=a.judge, limit=a.limit, workers=a.workers)
        exps = list(study.experiments()) if a.experiments == ["all"] else a.experiments
        unknown = set(exps) - set(study.experiments())
        if unknown:
            p.error(f"unknown experiment(s): {sorted(unknown)}")
        for e in exps:
            summ = st.run(e)
            for name, c in summ["conditions"].items():
                m = c["metrics"]
                show = {k: round(m[k]["mean"], 3) for k in
                        ("em", "f1", "hallucination_rate", "over_refusal", "retrieval_hit",
                         "misled_rate", "faithfulness") if k in m}
                print(f"  {e}/{name}: {show}")
        print(f"Results in {st.out_root}")
        return 0

    if a.cmd == "report":
        from analysis.report import build_report
        print(build_report(a.model))
        return 0


if __name__ == "__main__":
    sys.exit(main())
