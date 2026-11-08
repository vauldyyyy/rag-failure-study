# arXiv endorsement request — draft

**Status (8 Nov 2026):** paper complete; target venue CHOMPS 2027 @ EACL (deadline 15 Dec 2026). Under the ACL/ARR anonymity policy of 15 Jan 2024, non-anonymous preprints may be posted at any time (https://aclrollingreview.org/anonymity/); only the submitted PDF must be anonymous.

**Why this is urgent:** arXiv requires an endorsement from an established author
before a first-time submitter can post to a category (cs.CL). As of
**21 January 2026** arXiv no longer accepts institutional email as automatic
endorsement, so there is no way around asking a person. It is the only item on the
critical path that depends on someone else's inbox — send it before the results are
final, not after.

## Who to ask, in order of preference

1. **A professor in your department who posts to arXiv.** Check: search their name
   on arxiv.org. If they have recent cs.* submissions, they are almost certainly
   eligible. Easiest ask by far.
2. **Your project supervisor or a faculty member who knows your work**, even if
   their area is adjacent.
3. **An author of a paper you cite.** Cuconasu, Chen, Liu, Yoran and the others in
   `paper/refs.bib` receive endorsement requests routinely; it is a normal part of
   academic life, not an imposition. Use the version below with the citation line.

Do not mass-mail. One person at a time, wait ~4 days, then try the next.

---

## Draft — internal (professor at your institution)

> **Subject:** arXiv endorsement request (cs.CL) — Vauldan D'Souza, B.E. Computer Engineering
>
> Dear Professor [Name],
>
> I am a third-year B.E. Computer Engineering student at Goa College of
> Engineering. I have completed an empirical study on retrieval-augmented
> generation and would like to post it to arXiv, but as a first-time cs.CL
> submitter I need an endorsement from an established arXiv author.
>
> The study corrupts the evidence a RAG system retrieves (removing it, swapping in
> adjacent passages, or falsifying facts) and measures what the model does
> instead of answering correctly. Hallucination never rose; but with falsified
> evidence the model confidently repeated the falsehood about ten times more
> often, without abstaining. Hypotheses were preregistered, and the code and
> results are public: https://github.com/vauldyyyy/rag-failure-study
>
> If you are willing, arXiv will send you a short endorsement link; it takes a
> minute and does not make you responsible for the content. I am happy to send
> the draft first.
>
> Thank you for considering it.
>
> Vauldan D'Souza
> B.E. Computer Engineering, Goa College of Engineering · vauldandsouza@gmail.com

---

## Draft — external (an author you cite)

> **Subject:** arXiv endorsement request (cs.CL) — undergraduate preprint on RAG and falsified evidence
>
> Dear Dr [Name],
>
> I am a third-year undergraduate at Goa College of Engineering, India. I have
> written a short empirical study that builds on your work on [one specific
> finding of theirs, e.g. for ClashEval: "models adopting incorrect retrieved
> content over a correct prior more than 60% of the time"], which I cite.
>
> My study separates two failures that accuracy pools together: asserting
> something the evidence does not support, and faithfully repeating evidence that
> is false. Across nine corruption conditions the first never rose, while
> falsified evidence raised the second about tenfold with no rise in abstention.
> Hypotheses were preregistered; code and results are public:
> https://github.com/vauldyyyy/rag-failure-study
>
> As a first-time cs.CL submitter I need an endorsement to post the preprint. If
> you are willing, arXiv sends a short link; it takes a minute and carries no
> responsibility for the content. I am glad to send the draft first.
>
> Thank you for your time either way.
>
> Vauldan D'Souza
> Goa College of Engineering · vauldandsouza@gmail.com

---

## Before you send

- [ ] Fill in every bracket. An unedited template reads as spam.
- [ ] For the external version, name a **specific** finding of theirs. Generic
      flattery is obvious and counterproductive.
- [ ] Check the repo is public and the README renders — it is the only evidence
      they have that you are real.
- [ ] Do **not** claim the paper is finished if it is not. "I am writing this up"
      is fine and honest; endorsement is about your legitimacy as a submitter, not
      about the paper being complete.
- [ ] Keep it under ~200 words. Long requests get deferred.

## When you have the endorsement

It is per-category and does not expire, so getting it now costs nothing even if the
preprint is weeks away. Post when `verify_paper.py --strict` exits 0.
