# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary: a recruiter at a client company, doing real hiring.** They arrive with a
role to fill, describe it in their own words, and need a shortlist they can act
on — names, a way to reach them, and enough of each CV to decide. They are not
the person who built the system and have no reason to care how retrieval works.
Similarity scores, chunk counts and model ids are diagnostics to them, not
information.

**Secondary: prospective clients and employers, during a demo.** Showing the
39-candidate synthetic corpus to people is a confirmed, recurring workflow, not a
one-off. That audience is judging whether the system works and whether it was
built well.

**Operator: the repo owner**, who deploys, indexes, and diagnoses. The only person
who needs the raw numbers, and the reason they stay reachable rather than being
deleted.

## Product Purpose

Retrieval over the CV text that the rest of the hiring pipeline discards.
[hr-cv-pipeline](https://github.com/shibilshamz/hr-cv-pipeline) turns dropped PDFs
into 14 structured Google Sheet fields and throws the prose away. This service
keeps the prose and makes it searchable by meaning.

Success is a recruiter describing a role in plain English and getting back the
people who actually fit — including the candidate whose CV says "settlement
reconciliation service" and never once uses the word "payments". Failure is not a
wrong ranking; failure is a confident claim about experience the CV does not
contain.

## Positioning

Section-level semantic retrieval over CV prose, with the ranking grounded in
excerpts and made to cite them.

Two mechanisms a neighbouring product could not truthfully copy without doing the
same work:

- **Chunking on the CV's own structure**, then collapsing back to one row per
  person. A whole-CV embedding lands at the average of everything the document
  says and finds nobody; fixed windows cut job titles off their bullets. Splitting
  on headings and roles means every vector is one coherent claim, and over-fetch
  then collapse turns a list of passages back into a list of people.
- **A ranking that is allowed to say no.** The model sees only the retrieved
  excerpts, cites the candidate behind every claim, states plainly when an excerpt
  does not support a requirement, and refuses to infer anything from a name.

## Operating Context

CVs arrive on their own: a PDF dropped into a Google Drive Inbox is extracted by
the n8n pipeline and posted to `/index` on its way past. Nobody uploads anything
through this service.

The recruiter's actual loop is: write a brief → read the ranked candidates →
open the CVs worth reading → tick the ones worth contacting → export or copy the
shortlist → contact them outside this system. The work continues in email and on
the phone; this surface hands off to those and does not try to own them.

Two collections run side by side on one service: the live index of real
candidates, and a synthetic demo corpus used to show the system to prospects.
They are served by separate processes so a request can never reach the wrong one.

The host is a shared 3.8 GB VPS that also runs n8n, a trading dashboard and a
Flask job runner. There is no node toolchain on it.

## Capabilities and Constraints

Confirmed capabilities:

- Plain-English brief returns ranked candidates, each with a match strength, the
  section of their CV that matched, and their full extracted CV on demand.
- A shortlist that persists across several searches, exportable as a spreadsheet
  carrying name, email, phone, a CV link, the matched section, the similarity and
  the brief that found them.
- A written comparison of the retrieved candidates, grounded in their excerpts.
- An explicit "nobody here fits" state when the best match falls below the
  calibrated threshold.

Confirmed constraints:

- **The service holds extracted text, never the original document.** The PDF stays
  in the pipeline's Drive folder. A link to the original exists only for CVs
  indexed after the pipeline began sending one; everything earlier falls back to a
  rendered page of the extracted text.
- **No build step, no framework, no CDN, no webfont.** The UI is one HTML file read
  from disk per request. This follows from the host, and it is not a placeholder
  for a real front-end.
- Vector search is weak on exact strings — licence numbers, certification codes.
  Those need a hybrid keyword step that does not exist yet.
- Real candidate CVs are personal data. They are never committed, screenshotted,
  or used as demo material.

**Open decision — access model.** The service currently binds the Docker bridge and
is reachable only through an SSH tunnel, which means one operator on one desktop.
Whether it is eventually published behind authentication for client recruiters is
undecided. Future work must not assume either a shared URL or permanent privacy;
the choice changes whether sessions, roles and multi-device use are real concerns.

## Brand Commitments

Name: **CV Semantic Search**. Public repository, MIT licensed.

The documentation voice is a binding commitment because it is the product's
credibility: plain, specific, and explicit about limits. The README carries a
"What this is not" section; the chunking rationale states outright that its
numbers do not generalise beyond 39 synthetic candidates. Work on this product
states what it cannot do as readily as what it can.

## Evidence on Hand

- `corpus/synthetic/` — 39 candidates, generated by a seeded script, reproducible
  byte-for-byte. Deliberately built to break the chunker: three heading
  vocabularies, a CV with no headings, a near-duplicate person submitted by two
  agencies, and one candidate whose relevant skill is described in words no
  recruiter would type.
- `eval/briefs.yaml` and `eval/run_eval.py` — recall@5 1.00, MRR 1.00, negative
  control 1/1, separation between the worst genuine match (+0.413) and the best
  non-match (+0.307). Written before tuning.
- `docs/chunking.md` — the reasoning behind the one real design decision.
- One real CV in the live index.

Absences that must never be filled with invention: there are no customers, no
testimonials, no case studies, no press, no pricing, and no benchmark beyond the
39-candidate eval. The eval numbers describe that corpus and nothing larger.

## Product Principles

1. **The recruiter is not the builder.** Diagnostics stay reachable and stay out of
   the way. A number nobody can calibrate is not information.
2. **Never claim what the CV does not say.** Grounding is the product, not a
   safety feature bolted onto it. "Not stated in the CV" is a correct answer.
3. **Saying "nobody fits" is a feature.** A system that always returns five names
   has told the recruiter nothing.
4. **Demo data must never be mistakable for real people.** Two collections, two
   processes, and anything showing synthetic candidates says so on its face.
5. **The constraints are real, not temporary.** One file, no build, a small shared
   box. Work with them rather than around them.

## Accessibility & Inclusion

A hiring tool carries a specific obligation this one has already failed once and
now guards explicitly: an early run inferred a candidate's location from their
name. The ranking is now instructed never to infer nationality, location,
ethnicity, gender, age or religion from a name, and to answer "not stated in the
CV" when the excerpts do not say. This is a product requirement, not a prompt
detail — anything that ranks, filters, summarises or displays candidates inherits
it.
