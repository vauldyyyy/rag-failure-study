"""Package the paper as a flat zip that compiles on Overleaf as uploaded.

    python paper/make_overleaf_zip.py        # writes overleaf_upload.zip at the repo root

Contents: main.tex, refs.bib, generated/*.tex, the ACL style files, and every figure
main.tex includes (copied flat; \\graphicspath in main.tex also searches ./). Run
make_tables.py and verify_paper.py first so the zip carries checked numbers.
"""

import os
import re
import sys
import zipfile

PAPER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER)
OUT = os.path.join(ROOT, "overleaf_upload.zip")


def main():
    tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    m = re.search(r"\\graphicspath\{((?:\{[^}]*\})+)\}", tex)
    search = [""] + (re.findall(r"\{([^}]*)\}", m.group(1)) if m else [])

    files = {"main.tex": os.path.join(PAPER, "main.tex"),
             "refs.bib": os.path.join(PAPER, "refs.bib")}
    for name in ("acl.sty", "acl_natbib.bst"):
        path = os.path.join(PAPER, name)
        if not os.path.exists(path):
            sys.exit(f"{name} missing from paper/: the zip would build without the ACL template")
        files[name] = path
    gen = os.path.join(PAPER, "generated")
    for name in sorted(os.listdir(gen)):
        if name.endswith(".tex"):
            files[f"generated/{name}"] = os.path.join(gen, name)
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        found = [os.path.join(PAPER, d, fig) for d in search
                 if os.path.exists(os.path.join(PAPER, d, fig))]
        if not found:
            sys.exit(f"figure not found: {fig}")
        files[os.path.basename(fig)] = found[0]

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in files.items():
            z.write(src, arc)
    print(f"Wrote {OUT}")
    for arc in files:
        print(f"  {arc}")


if __name__ == "__main__":
    main()
