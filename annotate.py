"""Hand-labelling harness and automatic-label agreement.

This closes the two threats to validity that no amount of code can close on its
own: the judge shares a checkpoint with the generator, and abstention is detected
by a regex. Both are claims about label quality, and a claim about label quality
needs human labels to check against.

    python annotate.py sample --task judge      --n 150   # draw a stratified sample
    python annotate.py label  --task judge                # label it, resumable
    python annotate.py score  --task judge                # Cohen's kappa vs the judge

    python annotate.py sample --task abstention --n 100
    python annotate.py label  --task abstention
    python annotate.py score  --task abstention           # precision/recall of the rule

Samples are drawn with the global seed and saved, so the annotation set is fixed
before any label is seen — you cannot quietly re-draw until agreement looks good.
"""

import argparse
import glob
import json
import os
import random
import sys

import config

ANNO_DIR = os.path.join(config.BASE_DIR, "annotations")
os.makedirs(ANNO_DIR, exist_ok=True)

TASKS = {
    "judge": {
        "path": lambda: os.path.join(ANNO_DIR, "judge_labels.json"),
        "options": ["SUPPORTED", "UNSUPPORTED", "REFUSAL"],
        "keys": {"s": "SUPPORTED", "u": "UNSUPPORTED", "r": "REFUSAL"},
        "prompt": "Is every claim in the ANSWER stated in or entailed by the CONTEXT?",
    },
    "abstention": {
        "path": lambda: os.path.join(ANNO_DIR, "abstention_labels.json"),
        "options": ["ABSTAIN", "ASSERT"],
        "keys": {"a": "ABSTAIN", "s": "ASSERT"},
        "prompt": "Does the response DECLINE to answer, or does it assert something?",
    },
}


def _walk(node):
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "outcome" in item:
                yield item
            else:
                yield from _walk(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)


def collect_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "phase*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("manifest", {}).get("synthetic"):
            continue
        for row in _walk(payload.get("rows")):
            row["_source"] = os.path.basename(path)
            rows.append(row)
    return rows


def cmd_sample(args):
    task = TASKS[args.task]
    rows = collect_rows()
    if not rows:
        sys.exit(f"No non-synthetic results in {config.RESULTS_DIR}. Run the phases first.")

    if args.task == "judge":
        # Only items the judge actually ruled on are informative about the judge.
        pool = [r for r in rows if r.get("judge") in ("SUPPORTED", "UNSUPPORTED", "REFUSAL")]
    else:
        pool = [r for r in rows if r.get("prediction")]

    if not pool:
        sys.exit(f"No rows eligible for task '{args.task}'.")

    rng = random.Random(config.SEED)
    # Stratify by the automatic label so both classes are represented even when
    # one is rare; unstratified sampling of a 10% class wastes most of the budget.
    buckets = {}
    for r in pool:
        key = r.get("judge") if args.task == "judge" else bool(r.get("abstained"))
        buckets.setdefault(key, []).append(r)

    per_bucket = max(1, args.n // len(buckets))
    sample = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        rng.shuffle(bucket)
        sample.extend(bucket[:per_bucket])
    rng.shuffle(sample)
    sample = sample[:args.n]

    items = [{
        "qid": r.get("qid"),
        "condition": r.get("condition"),
        "question": r.get("question"),
        "prediction": r.get("prediction"),
        "gold_answers": r.get("gold_answers"),
        "is_answerable": r.get("is_answerable"),
        "auto_judge": r.get("judge"),
        "auto_abstained": r.get("abstained"),
        "auto_outcome": r.get("outcome"),
        "source": r.get("_source"),
        "human": None,
    } for r in sample]

    path = task["path"]()
    if os.path.exists(path) and not args.force:
        sys.exit(f"{path} already exists. Re-drawing the sample after seeing labels "
                 f"invalidates the agreement estimate. Pass --force if you are sure.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "seed": config.SEED,
                   "n": len(items), "items": items}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(items)} items to {path}\nNow run: python annotate.py label --task {args.task}")


def cmd_label(args):
    task = TASKS[args.task]
    path = task["path"]()
    if not os.path.exists(path):
        sys.exit(f"No sample at {path}. Run: python annotate.py sample --task {args.task}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    todo = [i for i in data["items"] if i["human"] is None]
    print(f"{len(todo)} unlabelled of {len(data['items'])}. "
          f"Keys: {', '.join(f'[{k}]={v}' for k, v in task['keys'].items())}, "
          f"[?]=skip, [q]=save and quit\n")

    for idx, item in enumerate(todo):
        print("=" * 72)
        print(f"({idx + 1}/{len(todo)})  {item['condition']}")
        print(f"Q: {item['question']}")
        if args.task == "judge":
            print(f"GOLD: {item['gold_answers'] or 'UNANSWERABLE'}")
        print(f"ANSWER: {item['prediction']}")
        print(f"\n{task['prompt']}")
        # The automatic label is deliberately not shown: seeing it would anchor
        # the annotator and inflate agreement.
        choice = input("> ").strip().lower()

        if choice == "q":
            break
        if choice == "?" or choice not in task["keys"]:
            if choice != "?":
                print("  unrecognised, skipping")
            continue
        item["human"] = task["keys"][choice]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    labelled = sum(1 for i in data["items"] if i["human"])
    print(f"\nSaved. {labelled}/{len(data['items'])} labelled.")


def cohens_kappa(a, b, categories):
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def cmd_score(args):
    task = TASKS[args.task]
    path = task["path"]()
    if not os.path.exists(path):
        sys.exit(f"No labels at {path}.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = [i for i in data["items"] if i["human"]]
    if not pairs:
        sys.exit("Nothing labelled yet.")

    if args.task == "judge":
        human = [i["human"] for i in pairs]
        auto = [i["auto_judge"] for i in pairs]
        cats = task["options"]
        agree = sum(h == a for h, a in zip(human, auto)) / len(pairs)
        kappa = cohens_kappa(human, auto, cats)
        report = {
            "task": "judge", "n": len(pairs),
            "raw_agreement": round(agree, 4),
            "cohens_kappa": None if kappa is None else round(kappa, 4),
            "confusion": {},
        }
        for h, a in zip(human, auto):
            report["confusion"].setdefault(f"human={h}", {}).setdefault(f"auto={a}", 0)
            report["confusion"][f"human={h}"][f"auto={a}"] += 1

        print(f"Judge vs human (n={len(pairs)})")
        print(f"  raw agreement : {agree:.3f}")
        print(f"  Cohen's kappa : {kappa:.3f}" if kappa is not None else "  kappa: n/a")
        print(_kappa_verdict(kappa))
    else:
        human = [i["human"] == "ABSTAIN" for i in pairs]
        auto = [bool(i["auto_abstained"]) for i in pairs]
        tp = sum(h and a for h, a in zip(human, auto))
        fp = sum((not h) and a for h, a in zip(human, auto))
        fn = sum(h and (not a) for h, a in zip(human, auto))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)
        kappa = cohens_kappa([str(h) for h in human], [str(a) for a in auto],
                             ["True", "False"])
        report = {
            "task": "abstention", "n": len(pairs),
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
            "cohens_kappa": None if kappa is None else round(kappa, 4),
            "false_positives": fp, "false_negatives": fn,
        }
        print(f"Abstention rule vs human (n={len(pairs)})")
        for key in ("precision", "recall", "f1", "cohens_kappa"):
            print(f"  {key:14s}: {report[key]}")
        print(f"  false positives (rule said abstain, human disagreed): {fp}")
        print(f"  false negatives (rule missed a real abstention):      {fn}")

    out = os.path.join(config.RESULTS_DIR, f"agreement_{args.task}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out} — cite this in the Limitations section.")


def _kappa_verdict(k):
    if k is None:
        return ""
    if k >= 0.8:
        return "  -> strong agreement; the automatic judge is reportable as-is."
    if k >= 0.6:
        return "  -> substantial agreement; report kappa alongside every judge-based number."
    return ("  -> weak agreement. Do not report judge-based hallucination rates as "
            "primary results; lead with the rule-based unanswerable rate instead.")


def main():
    p = argparse.ArgumentParser(description="Human annotation and agreement")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sample", cmd_sample), ("label", cmd_label), ("score", cmd_score)):
        sp = sub.add_parser(name)
        sp.add_argument("--task", required=True, choices=list(TASKS))
        sp.add_argument("--n", type=int, default=150)
        sp.add_argument("--force", action="store_true")
        sp.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
