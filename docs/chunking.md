# Why chunk a CV by section

This is the only real design decision in the project. Everything else — the
vector store, the API, the ranking prompt — is wiring that could be swapped for
an equivalent without changing the results much. This one changes them a lot.

## The problem chunking exists to solve

An embedding compresses a passage into a single point in 384-dimensional space.
Two passages about the same thing land near each other; two passages about
different things land apart. That is the whole mechanism.

The failure mode follows directly. Feed the model an entire CV and the point
lands at the *average* of everything the document says. A three-page CV covering
React Native, warehouse logistics and a chemistry degree embeds to somewhere
near none of them — a point in the middle of a space where nobody is looking.
This is **dilution**, and it gets worse the more varied and the longer the
document is. CVs are unusually bad for it: a CV is a deliberately multi-topic
document, and the whole point of searching one is to find a single narrow claim
buried inside it.

## Three options

### One chunk per CV — rejected

Simplest to build and simplest to explain. One vector per person, no collapsing
needed at retrieval, the index is exactly as big as the candidate list.

It fails on precision for the reason above. A query about one narrow skill
competes against every unrelated paragraph sharing that vector. And it fails
worst on exactly the candidates a recruiter most wants to find: the generalist
with eight years of varied work, whose CV averages out to nothing in particular.

### Fixed-size windows with overlap — rejected

The generic default, and the right answer for prose — manuals, policy documents,
transcripts, anything with no structure of its own. Take 500 tokens, slide, add
50 tokens of overlap so a sentence spanning a boundary survives in one piece.

On a CV it throws away structure the document already carries. A 500-token
window cuts mid-sentence, splits a job title from the bullets underneath it, and
merges the tail of one role with the head of the next. The overlap patches the
symptom rather than the cause. You end up reconstructing sections badly when the
document had them all along.

### One chunk per section — chosen

Split on the headings a CV already carries, and split the experience section
further, one chunk per role.

Every vector then represents one coherent claim: *this* summary, *this* job,
*this* skills list. A query for "Kubernetes" hits the skills section or the role
where it was used, not a blurred average of a whole career. The section label
comes along as metadata, so the recruiter is told *why* someone matched — "Senior
Backend Engineer, Emirates NBD" — rather than just that they did.

## What it costs, and how that is handled

Section chunking has one direct consequence: **one candidate now produces five to
seven vectors instead of one.** A top-20 vector search can return the same person
five times, which is useless to a recruiter who asked for five candidates.

So retrieval over-fetches and then collapses:

```python
hits = col.query(query_embeddings=..., n_results=k * 4)
best = {}
for doc, meta, dist in zip(...):
    cid = meta["candidate_id"]
    if cid not in best or dist < best[cid]["distance"]:
        best[cid] = {...}
return sorted(best.values(), key=lambda r: r["distance"])[:k]
```

Fetch four times as many chunks as you need candidates, keep each person's single
best-scoring chunk, and remember which section it came from. That section is the
match reason shown to the recruiter.

## Two implementation details that are load-bearing

**A heading must occupy its own line.** The obvious regex — match a heading word
at the start of a line — also matches "Experience in payment systems includes…"
and splits a CV mid-paragraph. The damage is invisible: the chunks look
plausible, nothing errors, and retrieval just gets quietly worse. `chunker.py`
requires the whole line to be the heading and nothing else.

**Roles are marked by dates, but do not start with them.** The reliable signal
for "a new job starts here" is a date range. The line above it is the job title
and employer. Cutting at the date line alone strips every title from the chunk it
belongs to and glues the section heading onto the first role, so the boundary
walks back one line.

## Does it work

`eval/briefs.yaml` includes a brief whose correct answer shares no vocabulary
with it at all. The corpus contains no occurrence of "payments" or "API"; the
target CV says "settlement reconciliation", "card movements" and "issuer
statements". Keyword search returns nothing. Section-level retrieval returns him
first, matched on his summary section, at a cosine score of +0.41 against a best
non-match of +0.31.

That gap is the argument for the whole approach, and it is why the eval set is
in the repo rather than a paragraph claiming the system is accurate.

## What this is not

One retrieval system over one document type, evaluated on 39 synthetic
candidates. The design rationale generalises; the numbers do not. At a few
thousand real CVs the next problems are a hybrid keyword-plus-vector retrieval
step for exact terms like certifications and licence numbers, and a reranker
between retrieval and generation. Neither is needed at this size.
