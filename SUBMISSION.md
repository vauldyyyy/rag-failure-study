# Submission checklist

Target: **CHOMPS 2027** — 2nd Workshop on Confabulation, Hallucinations &
Overgeneration in Multilingual and Practical Scenarios, at **EACL 2027**
(Athens, 9–14 March 2027). Source: https://chomps-workshop.github.io/ (read 8 Nov 2026).

| Date (AoE, 23:59) | What |
|---|---|
| **15 Dec 2026** | Paper submission deadline |
| 22 Dec 2026 | ARR commitment deadline (not needed: we submit directly) |
| 5 Jan 2027 | Notification |
| 19 Jan 2027 | Camera-ready |

Format: **short paper, up to 4 pages of content**, official ACL template, PDF.
Archival (in the proceedings) or non-archival — choose **archival**.
Accepted papers may add one page.

## 1. Build the PDF (Overleaf, ~10 minutes)

1. Download `overleaf_upload.zip` from the repo root (it contains `main.tex`,
   `refs.bib`, `generated/`, the figure, and the official ACL style files).
2. overleaf.com → **New Project → Upload Project** → choose the zip.
3. Menu → Compiler: **pdfLaTeX** → **Recompile**.
4. Check the page count. Content (everything before References) must fit in
   **4 pages**. Limitations, Ethics and References come after and do not count
   toward the 4 pages under ACL rules. If it runs over, trim Section 5 first.

The template is in **review** mode (`\usepackage[review]{acl}`), which hides the
author block and adds line numbers, as the submission requires. For arXiv and the
camera-ready, change `review` to `final`.

## 2. Anonymity (ACL/ARR policy of 15 Jan 2024)

- The submitted PDF must be anonymous. Review mode handles the author block. The
  paper does not link the GitHub repository; keep it that way in the submission.
  If you want reviewers to see code, upload an anonymised copy (e.g.
  anonymous.4open.science) and link that instead.
- Posting a **non-anonymous arXiv preprint is allowed at any time**
  (https://aclrollingreview.org/anonymity/). Avoid heavy promotion while under
  review; that is advice, not a rule.

## 3. Submit on OpenReview

1. Create an OpenReview profile with your college email if you can (profiles with
   an institutional email are activated faster; a Gmail profile can take up to two
   weeks to be moderated — **do this now**).
2. Direct submission page:
   https://openreview.net/group?id=eacl.org/EACL/2027/Workshop/CHOMPS
3. **Reciprocal reviewing**: every direct submission must name an eligible
   reviewer who will review 1–2 papers. Non-compliance can lead to desk rejection.
   Check whether an undergraduate first author qualifies; if not, ask a faculty
   co-author or mentor to be the designated reviewer.
4. **Generative-AI disclosure** is required (ARR transparency rules). State that
   an AI assistant was used for code review, writing assistance and literature
   checks, and that all experiments, numbers and claims were verified by the
   author. Be accurate; this is checked.
5. Note: submissions with a high-school student author are not eligible
   (does not apply here).

## 4. arXiv (any time)

1. Get an endorsement for **cs.CL** — see `ENDORSEMENT_EMAIL.md`; one person at a
   time, wait about four days before trying the next.
2. Switch the template to `final`, recompile, upload source + PDF.
3. Primary category cs.CL; cross-list cs.IR and cs.AI.

## 5. Before you press submit

```bash
python paper/make_tables.py --dataset squad
python verify_paper.py          # must print 0 failure(s)
python tests/test_offline.py    # must print ALL PASS
```
