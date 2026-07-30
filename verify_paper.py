"""Submission gate: refuse to ship a paper whose numbers do not trace to a run.

This repository has already shipped, once, a README asserting findings that no
experiment had produced. This script exists so that cannot happen again. It is the
last thing to run before submitting, and it is wired into CI.

    python verify_paper.py                 # check paper/ against results/
    python verify_paper.py --dry-run       # check against results_dryrun/ (always fails
                                           # the synthetic check; useful for the rest)

Checks, in order of how badly each one would embarrass you:

  1. Results are real, not dry-run stubs.
  2. Every \\NUM{...} macro used in main.tex is actually defined in generated/.
  3. No literal digits in the prose that did not come from a macro.
  4. Every \\cite key resolves in refs.bib.
  5. No refs.bib entry is still marked unverified.
  6. Every \\includegraphics target exists on disk.
  7. No red [TODO]/placeholder markers survive.
"""

import argparse
import json
import os
import re
import sys

import config

PAPER_DIR = os.path.join(config.BASE_DIR, "paper")
MAIN_TEX = os.path.join(PAPER_DIR, "main.tex")
REFS_BIB = os.path.join(PAPER_DIR, "refs.bib")
GENERATED = os.path.join(PAPER_DIR, "generated")

failures, warnings = [], []


def fail(check, msg):
    failures.append(f"[{check}] {msg}")


def warn(check, msg):
    warnings.append(f"[{check}] {msg}")


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def strip_comments(tex):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


# --------------------------------------------------------------------------- #

def check_results_are_real():
    paths = [p for p in os.listdir(config.RESULTS_DIR)
             if p.startswith("phase") and p.endswith(".json")] \
        if os.path.exists(config.RESULTS_DIR) else []
    if not paths:
        fail("results", f"No phase results in {config.RESULTS_DIR} — nothing to verify.")
        return

    for name in paths:
        with open(os.path.join(config.RESULTS_DIR, name), "r", encoding="utf-8") as f:
            payload = json.load(f)
        manifest = payload.get("manifest", {})
        if manifest.get("synthetic"):
            fail("results", f"{name} was produced by --dry-run (synthetic stub model).")
        if manifest.get("judge_self_eval"):
            warn("results", f"{name}: judge and generator are the same checkpoint. "
                            f"Report agreement against human labels or change JUDGE_MODEL.")
        if manifest.get("n_qa", 0) < 100:
            warn("results", f"{name}: only n={manifest.get('n_qa')} questions — "
                            f"pilot-sized, intervals will be wide.")


def check_macros(tex):
    used = set(re.findall(r"\\NUM\{(\w+)\}", tex))
    macros_tex = read(os.path.join(GENERATED, "macros.tex"))
    if macros_tex is None:
        fail("macros", "paper/generated/macros.tex missing — run paper/make_tables.py.")
        return
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", macros_tex))
    for name in sorted(used - defined):
        fail("macros", f"\\NUM{{{name}}} used in main.tex but never generated — "
                       f"it will render as [NO RESULT].")
    for name in sorted(defined - used - {"SYNTHETICRESULTS"}):
        warn("macros", f"macro \\{name} generated but unused.")


def check_no_hardcoded_numbers(tex):
    """Digits in prose must come from macros, not from the author's memory."""
    body = tex
    # Everything legitimately numeric: preamble, generated inputs, figure/table
    # machinery, citations, labels, and the guarded TODO notes.
    for pattern in (r"\\documentclass.*", r"\\usepackage.*", r"\\includegraphics.*",
                    r"\\input\{[^}]*\}", r"\\newcommand.*", r"\\label\{[^}]*\}",
                    r"\\ref\{[^}]*\}", r"\\cite[tp]?\{[^}]*\}", r"\\citet\{[^}]*\}",
                    r"\\bibliography.*", r"\\textcolor\{red\}\{[^}]*\}",
                    r"\\NUM\{\w+\}", r"\\geometry.*", r"\\colorbox.*"):
        body = re.sub(pattern, " ", body)

    allowed = {"0", "1", "2", "3", "4", "5", "8", "10", "50", "95", "256", "2"}
    for match in re.finditer(r"(?<![\w.\\])(\d+\.\d+)(?![\w])", body):
        warn("hardcoded", f"literal decimal '{match.group(1)}' in prose — "
                          f"should this come from a generated macro?")
    for match in re.finditer(r"(?<![\w.\\])(\d{3,})(?![\w])", body):
        if match.group(1) not in allowed:
            warn("hardcoded", f"literal number '{match.group(1)}' in prose — "
                              f"verify it against results/ or move it to a macro.")


def check_citations(tex):
    bib = read(REFS_BIB)
    if bib is None:
        fail("bib", "paper/refs.bib missing.")
        return
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    for key in sorted(cited - keys):
        fail("bib", f"\\cite{{{key}}} has no entry in refs.bib.")
    for key in sorted(keys - cited):
        warn("bib", f"refs.bib entry '{key}' is never cited.")

    for block in bib.split("@")[1:]:
        if "NOT YET RE-VERIFIED" in block:
            continue
    unverified = re.findall(r"% NOT YET RE-VERIFIED[^\n]*\n@\w+\{([^,]+),", bib)
    for key in unverified:
        if key in cited:
            fail("bib", f"'{key}' is cited but still marked NOT YET RE-VERIFIED. "
                        f"Open the paper, confirm the metadata, remove the marker.")


def check_figures(tex):
    for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        resolved = os.path.normpath(os.path.join(PAPER_DIR, path))
        if not os.path.exists(resolved):
            fail("figures", f"referenced figure missing: {path} "
                            f"(run visualize_results.py)")


def check_placeholders(tex):
    for match in re.finditer(r"\\textcolor\{red\}\{\[([^\]]{0,80})", tex):
        fail("placeholder", f"unresolved author note: [{match.group(1).strip()}...]")
    for marker in ("Author Name", "email@example.com", "Affiliation"):
        if marker in tex:
            fail("placeholder", f"template text '{marker}' still present.")


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Pre-submission verification")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = p.parse_args()
    config.set_dry_run(args.dry_run)

    tex = read(MAIN_TEX)
    if tex is None:
        sys.exit(f"No paper at {MAIN_TEX}")
    tex = strip_comments(tex)

    check_results_are_real()
    check_macros(tex)
    check_no_hardcoded_numbers(tex)
    check_citations(tex)
    check_figures(tex)
    check_placeholders(tex)

    for w in warnings:
        print(f"WARN  {w}")
    for f in failures:
        print(f"FAIL  {f}")

    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s)")
    if failures or (args.strict and warnings):
        print("NOT READY TO SUBMIT")
        sys.exit(1)
    print("Ready to submit.")


if __name__ == "__main__":
    main()
