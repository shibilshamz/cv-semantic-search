# 🔍 CV Semantic Search

Describe a role in plain English, get back the candidates who actually fit —
including the one whose CV says *"settlement reconciliation service"* and never
uses the word *"payments"*.

A retrieval layer over the CVs processed by
[hr-cv-pipeline](https://github.com/shibilshamz/hr-cv-pipeline), which turns
dropped PDFs into structured Google Sheet rows. That pipeline keeps 14 fields and
throws the CV text away. This keeps the text and makes it searchable by meaning.

```
brief: "backend engineer comfortable with payments APIs, based in the Gulf"

1. Ahmed Al-Rashid            +0.413  [Professional Summary]
2. Fatima Zahra Bennani       +0.391  [Technical Skills]
3. Hassan Nakamura            +0.330  [Core Competencies]
```

Neither *"payments"* nor *"API"* appears anywhere in Ahmed's CV, or anywhere in
the corpus. He is found on meaning alone.

---

## How it works

```
INDEX — once per CV
   CV text ──▶ split into sections ──▶ embed (384-d) ──▶ Chroma
                                                           │
RETRIEVE + GENERATE — per query                            │
   job brief ──▶ embed ──▶ nearest chunks ──▶ collapse ────┘
                            (k × 4)          by candidate
                                                  │
                                                  ▼
                                        top 5 ──▶ Claude Haiku ──▶ ranked
                                                  ranks + cites      answer
```

The over-fetch-then-collapse step is the part that matters. Chunking is
section-level, so one strong candidate can legitimately occupy five of the top
ten slots with five different sections of their own CV. Retrieval pulls `k × 4`
chunks, keeps each person's single best-scoring one, and returns a list of
*people* rather than a list of passages.

The model never sees the other 189 chunks. That is the whole economy of the
thing: ranking 5 candidates costs a fraction of a cent, ranking 500 CVs would not
be worth doing.

---

## Design decisions

| Component | Choice | Why |
|---|---|---|
| Chunking | **one chunk per CV section**, experience split per role | A CV is a multi-topic document; a whole-document embedding averages React Native, warehouse logistics and a chemistry degree into a point near none of them. [Full rationale →](docs/chunking.md) |
| Embeddings | **all-MiniLM-L6-v2 via ONNX Runtime** | Same model and same 384 dimensions as sentence-transformers at roughly a tenth of the footprint. The default torch wheel drags ~2.5 GB of CUDA packages onto a 3.8 GB VPS with no GPU. |
| Vector store | **Chroma**, pinned, in Docker | Already have Docker up for n8n. Bind-mounted so the index survives a container recreate. Client and server pinned to the same version. |
| Ranking | **Claude Haiku 4.5**, pinned ID | Already in the stack. Pinned rather than the moving alias, because a reproducible eval is the point. |
| Identity | **email as `candidate_id`** | Already the dedup key in the n8n pipeline, so both systems agree on who someone is without a mapping table. |

Chunk IDs are `{email}:{i}`, which makes re-indexing an upsert over the same
rows. A candidate submitted twice by two different agencies collapses to one
entry at index time, before retrieval ever sees them.

---

## Results

`eval/run_eval.py` against the 39-candidate synthetic corpus, retrieval only —
no model call, so it is free, deterministic, and a regression points at the
chunker rather than at model variance.

```
  recall@5          1.00  (7/7)
  MRR               1.00
  negative control  1/1
  separation        worst genuine +0.413 vs best non-match +0.307
```

The eval set was written before any tuning. It deliberately includes the three
cases that are easy to get wrong:

| Case | What it proves |
|---|---|
| A brief sharing **no vocabulary** with its answer | Semantic retrieval is doing something keyword search cannot |
| A brief with **no good match** (`veterinary surgeon, equine specialist`) | The system reports a weak result instead of forcing five names |
| A candidate present in the corpus **twice**, from two agencies | Collapse-by-candidate works |

Plus one CV with no section headings at all, which exercises the chunker's
fallback path and still ranks first for its brief.

**MRR of 1.00 means every expected candidate ranked first.** On 39 synthetic
candidates. These numbers say the pipeline is wired correctly, not that it would
hold at five thousand real CVs — see [the last section of
docs/chunking.md](docs/chunking.md) for what would need to change at that size.

---

## Running it

Needs Python 3.10+. No Docker required for local use — set `CHROMA_PATH` and
Chroma runs embedded.

```bash
pip install -r requirements.txt

python corpus/generate.py                        # 40 synthetic CVs, seeded
export CHROMA_PATH=./.chroma-dev                 # local mode, no Docker
python indexer.py --corpus corpus/synthetic
python eval/run_eval.py

python search.py "data engineer who has built streaming pipelines"
```

Each module runs its own tests — `python chunker.py`, `python embedder.py` — in
the style of the sibling repos.

Then open <http://127.0.0.1:5680/> for the search page:

```bash
uvicorn api:app --host 127.0.0.1 --port 5680
```

For the API, the VPS deployment, and day-to-day use (the UI over an SSH tunnel,
one-command searches, clearing the demo corpus before going live), see
[deploy/DEPLOY.md](deploy/DEPLOY.md).

```bash
curl -X POST localhost:5680/search -H 'content-type: application/json' \
  -d '{"text":"critical care nurse, DHA licensed","k":5}'
```

| Route | Purpose |
|---|---|
| `GET /` | The search page — a brief in, ranked candidates out |
| `GET /health` | Liveness, plus how many chunks are actually indexed |
| `POST /index` | `{candidate_id, name, text}` — what the n8n pipeline calls |
| `POST /search` | `{text, k, explain}` — `explain: false` skips the Claude call |

---

## Grounding

The ranking prompt is constrained to the retrieved excerpts:

> Use ONLY the excerpts provided. Cite the `[candidate_id]` behind every claim.
> If an excerpt does not support a requirement, say so explicitly. Never infer
> experience that is not written down.

Without that instruction the model fills gaps from its own priors and describes
experience the candidate does not have — confidently, in fluent prose, attributed
to a real person about to be put in front of a client. Everything else in
`search.py` is plumbing; that paragraph is what makes the output safe to act on.

The `weak` flag in the response is the same idea in a field you can branch on:
when the best match scores below the calibrated threshold, the caller is told so
rather than having to infer it from prose.

---

## Deployment shape

Runs on the same Ubuntu VPS as the pipeline that feeds it.

| Service | Port | Bound to | Why |
|---|---|---|---|
| Chroma | 8001 | `127.0.0.1` | 8000 is taken by another service on this box. Only the API talks to it. |
| cv-search API | 5680 | `172.17.0.1` | The docker0 bridge address — listens on the bridge only, never on the public interface |

Binding the API to the bridge rather than `0.0.0.0` is deliberate. A service on
`0.0.0.0` listens on every interface including the public one, which leaves the
firewall as the only thing between a candidate database and the open internet.
Binding the bridge address means an accidental firewall change cannot expose it,
because it was never listening there.

Reaching it *from* a container needs one firewall rule on top, because
container-to-host traffic arrives through the host `INPUT` chain where ufw's
default policy is `DROP`:

```bash
ufw allow in on docker0 from 172.17.0.0/16 to 172.17.0.1 port 5680 proto tcp
```

Scoped to the interface, the bridge subnet and the single port — as opposed to
`ufw allow 5680`, which would open it to the internet. Confirm from another
machine rather than trusting `ufw status`.

Secrets live in a `chmod 600` `.env` read through `python-dotenv`, not inline in
the Supervisor config.

---

## Stack

Python · ONNX Runtime · Chroma · Claude Haiku 4.5 · FastAPI · Docker · Supervisor · Ubuntu VPS

## Privacy

The committed corpus is **synthetic** — generated by `corpus/generate.py`, seeded
and reproducible. Real candidate CVs are personal data: they are never committed,
and `.gitignore` blocks `corpus/real/` and every PDF.

## License

MIT — see [LICENSE](LICENSE).
