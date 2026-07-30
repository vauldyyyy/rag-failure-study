# arXiv endorsement request — draft

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

> **Subject:** arXiv endorsement request (cs.CL) — [your name], [your programme]
>
> Dear Professor [Name],
>
> I am a second-year undergraduate in [programme] at [institution]. I have written
> up an empirical study on retrieval-augmented generation and would like to post it
> to arXiv as a preprint, but as a first-time submitter to cs.CL I need an
> endorsement from an established arXiv author.
>
> The study asks whether a RAG system, when its retrieval degrades, refuses to
> answer or instead asserts something unsupported — and whether that failure can be
> predicted from retrieval scores before the generator is called. The code,
> preregistered hypotheses and results are public here:
> https://github.com/vauldyyyy/rag-failure-study
>
> If you are willing, arXiv will send you a short endorsement link; the process
> takes a minute and does not make you responsible for the content. I am happy to
> send the draft first if you would prefer to read it before deciding.
>
> Thank you for considering it.
>
> [Your name]
> [Programme, institution] · [email]

---

## Draft — external (an author you cite)

> **Subject:** arXiv endorsement request (cs.CL) — undergraduate preprint on RAG failure modes
>
> Dear Dr [Name],
>
> I am a second-year undergraduate at [institution]. I have written a short
> empirical study of RAG failure behaviour that builds directly on your work on
> [their paper, one specific point — e.g. "the finding that related-but-irrelevant
> documents degrade accuracy more than random ones"], which I cite.
>
> The study measures whether failures are *loud* (the model abstains) or *silent*
> (it asserts something the retrieved context does not support), and tests whether
> retrieval scores predict the silent case before generation. Hypotheses were
> preregistered before data collection; code and results are public:
> https://github.com/vauldyyyy/rag-failure-study
>
> As a first-time cs.CL submitter I need an endorsement to post the preprint. If
> you are willing, arXiv sends a short link — it takes a minute and carries no
> responsibility for the content. I am glad to send the draft first.
>
> Thank you for your time either way.
>
> [Your name]
> [Programme, institution] · [email]

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
