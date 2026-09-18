"""HTTP interface.

    uvicorn api:app --host $API_HOST --port $API_PORT

Bound by default to 172.17.0.1 -- the docker0 bridge address -- not 0.0.0.0 and
not 127.0.0.1:

  127.0.0.1  invisible to the n8n container, which is the one client that needs
             to reach POST /index
  0.0.0.0    listens on every interface including the public one, so the
             firewall becomes the only thing standing between a candidate
             database and the internet. That is how job-runner on 5679 is set
             up, and it is the part of that setup worth not copying.
  172.17.0.1 listens only on the bridge. Not bound to the public interface at
             all, so an accidental firewall change cannot expose it.

Reaching it from a container additionally needs a firewall rule, because
container-to-host traffic arrives through the host INPUT chain and ufw's default
policy there is DROP. The rule is scoped to the interface, the bridge subnet and
this one port:

    ufw allow in on docker0 from 172.17.0.0/16 to 172.17.0.1 port 5680 proto tcp

That grants exactly what n8n needs and nothing else -- unlike `ufw allow 5680`,
which would open the port to the world. Verified from off-box, since `ufw status`
is not evidence of anything.

Set API_HOST=127.0.0.1 in .env for local development.
"""

import html
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

import store
from chunker import LABEL_CHARS
from contact import phone_from
from indexer import index_cv
from search import MIN_SCORE, retrieve, search

logger = logging.getLogger("cv-search")

app = FastAPI(
    title="CV Semantic Search",
    description="Retrieval over the CVs processed by the HR CV Pipeline.",
    version="1.0.0",
)


class Brief(BaseModel):
    text: str = Field(..., min_length=3, description="the role, in plain English")
    k: int = Field(5, ge=1, le=20)
    explain: bool = Field(True, description="false skips the Claude ranking call")


class Candidate(BaseModel):
    # An email, because that is the dedup key the n8n pipeline already uses.
    # Validated rather than trusted: an extraction that finds no email would
    # otherwise index the candidate under the id "", and every subsequent
    # email-less CV would overwrite the last one under that same key. The
    # pipeline already skips these -- a CV with no email is treated as a
    # duplicate and never reaches the sheet -- so rejecting them here keeps the
    # two systems agreeing on who counts as a person.
    candidate_id: str = Field(..., min_length=3, description="the candidate's email")
    name: str = ""
    text: str = Field(..., min_length=1, description="extracted CV text")
    source: str = ""
    # Where the original document lives, if the caller knows. The pipeline has
    # this -- a Drive node returns webViewLink alongside the file -- it just has
    # not been sending it. Optional, because every CV already indexed predates
    # this field and must keep working without one.
    source_url: str = ""

    @field_validator("candidate_id")
    @classmethod
    def _looks_like_an_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError("candidate_id must be an email address")
        return v


UI = Path(__file__).parent / "ui.html"


def _is_synthetic() -> bool:
    """Does this collection hold generated people rather than real ones?

    Read from the rows, not from the collection name. The UI's "not real people"
    banner used to key off /demo|test|sample/ against the name, so the local dev
    corpus -- named "cvs" -- served forty invented candidates, complete with
    phone numbers and visa status, wearing no warning. A config value is not a
    safety mechanism.

    A row must say so. Unknown does NOT count as synthetic, and that direction is
    chosen rather than inherited: the live index is fed by the CV pipeline, so
    flagging real candidates as generated has its own cost -- a recruiter who
    discounts a real person, or learns to ignore the banner because it cries
    wolf. Every CV indexed from now on carries the field either way, since
    index_cv writes False by default.

    The gap that leaves is rows written before the field existed. They read as
    real. If a generated corpus predates this change, reindex it -- for
    corpus/synthetic that is one command and the flag is inferred:

        CHROMA_COLLECTION=<name> .venv/bin/python indexer.py --corpus corpus/synthetic
    """
    try:
        col = store.get_collection()
        if col.count() == 0:
            return False
        return bool((col.get(where={"synthetic": True}, limit=1, include=[]) or {}).get("ids"))
    except Exception:
        return False


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """The search page.

    Served from this app rather than hosted separately because the API binds to
    the docker bridge: any other origin would need the service exposed publicly
    or CORS opened, and a static file on the same origin needs neither.

    Read per request, not cached at import, so editing ui.html on the box shows
    up on refresh without a supervisorctl restart. One file read on a page load
    that is about to make a Claude call is not the cost worth optimising.
    """
    try:
        return UI.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(404, "ui.html is missing; the JSON API still works")


@app.get("/health")
def health():
    """Liveness plus the one fact worth knowing: is anything actually indexed."""
    try:
        count = store.get_collection().count()
    except Exception as exc:
        raise HTTPException(503, f"vector store unreachable: {type(exc).__name__}")
    # The collection name is here because two instances of this service can point
    # at the same Chroma and different collections -- a demo corpus alongside the
    # real one. Two identical-looking pages, one holding real candidate data, is
    # exactly the confusion worth spending a field on.
    return {"status": "ok", "store": store.describe(), "chunks": count,
            "collection": store.COLLECTION,
            "synthetic": _is_synthetic(),
            "model": store.CLAUDE_MODEL, "min_score": MIN_SCORE}


def _strip_heading(text: str, section: str) -> str:
    """Drop a chunk's first line when it just repeats the section label.

    The chunker splits on heading lines and keeps them, so every chunk begins
    with its own title. Anything rendering the label above the text therefore
    says it twice. Done here rather than in the page so the JSON and the HTML
    view agree, and there is one copy of the rule.

    Compared exactly, case- and punctuation-insensitively, so a section whose
    first line merely resembles its label keeps its text.
    """
    lines = text.split("\n")
    first = lines[0].strip().rstrip(":\u2013\u2014-").strip()
    label = (section or "").strip()
    # A label that hit the chunker's cap is a prefix of the line it came from,
    # so exact comparison misses it. Relaxed to a prefix test only at exactly
    # the cap -- a shorter label still has to match in full, which is what keeps
    # "Skills" from eating a line beginning "Skills in distributed tracing...".
    truncated = len(label) == LABEL_CHARS and first.casefold().startswith(label.casefold())
    if first and (first.casefold() == label.casefold() or truncated):
        return "\n".join(lines[1:]).lstrip("\r\n")
    return text


@app.get("/cv/{candidate_id}", response_class=HTMLResponse, include_in_schema=False)
def cv_page(candidate_id: str):
    """One candidate's CV as a readable page, so a link can point somewhere.

    This is what the shortlist export links to when the pipeline has not supplied
    a URL for the original document -- which today is every candidate. It is the
    extracted text, and the page says so rather than letting a reader assume they
    are looking at the file the candidate sent.

    Worth being clear about what the link is worth: it resolves on whatever host
    is serving this API, which for a recruiter means only while their SSH tunnel
    is open. Mailing the spreadsheet to someone else does not carry the CV with
    it. A real, shareable link needs source_url, and that has to come from the
    pipeline.
    """
    data = candidate_detail(candidate_id)
    esc = html.escape

    blocks = "".join(
        f"<section><h2>{esc(sec['section']) or 'Section'}</h2>"
        f"<pre>{esc(sec['text'])}</pre></section>"
        for sec in data["sections"]
    )
    phone = (f" &middot; <a href=\"tel:{esc(data['phone'].replace(' ', ''))}\">"
             f"{esc(data['phone'])}</a>") if data["phone"] else ""
    original = (f"<p class=\"orig\"><a href=\"{esc(data['source_url'])}\" "
                f"rel=\"noopener noreferrer\" target=\"_blank\">Open the original document</a></p>"
                ) if data["source_url"] else ""

    return f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(data['name']) or esc(candidate_id)} &mdash; CV</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ max-width: 46rem; margin: 0 auto; padding: 2rem 1.25rem 4rem;
         font: 15px/1.6 ui-sans-serif, system-ui, "Segoe UI", sans-serif; }}
  h1 {{ font-family: ui-serif, Georgia, serif; font-size: 1.5rem; margin: 0 0 .2rem; }}
  .meta {{ color: #6b6b66; font-size: .9rem; margin: 0 0 1.5rem; }}
  .meta a {{ color: inherit; }}
  h2 {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .05em;
       color: #6b6b66; margin: 1.6rem 0 .3rem; }}
  pre {{ margin: 0; font: inherit; white-space: pre-wrap; word-wrap: break-word; }}
  .orig {{ margin: 1.5rem 0 0; }}
  footer {{ margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #ddd;
           color: #6b6b66; font-size: .82rem; }}
  @media print {{ footer, .orig {{ display: none }} }}
</style>
<h1>{esc(data['name']) or 'Name not on file'}</h1>
<p class="meta"><a href="mailto:{esc(data['candidate_id'])}">{esc(data['candidate_id'])}</a>{phone}</p>
{blocks}
{original}
<footer>Text extracted from {esc(data['source']) or 'the uploaded CV'} when it was
indexed. This is what the search reads, not the original file.</footer>
"""


@app.get("/candidate/{candidate_id}")
def candidate_detail(candidate_id: str):
    """Every indexed chunk for one candidate, reassembled in document order.

    The search result gives a recruiter a name, a score and the single section
    that matched. That is enough to rank people and not enough to decide about
    one, so this returns the CV behind the match.

    It is the *extracted text*, not the original file. For a real candidate the
    PDF goes to the pipeline's Drive folder and never reaches this service --
    only the text does. Saying so in the payload beats letting a caller assume
    it is holding the document the candidate actually sent.

    Chunks are ordered by the integer suffix of their id, not by the order
    Chroma returns them in, which is unspecified. Sorting the raw id strings
    would put ':10' before ':2'.
    """
    candidate_id = candidate_id.strip().lower()
    try:
        got = store.get_collection().get(
            where={"candidate_id": candidate_id},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        raise HTTPException(503, f"vector store unreachable: {type(exc).__name__}")

    if not got.get("ids"):
        raise HTTPException(404, "no such candidate in this index")

    def order(item):
        chunk_id = item[0]
        tail = chunk_id.rsplit(":", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=order)
    first = rows[0][2]

    sections = [{"section": m.get("section", ""),
                 "text": _strip_heading(d, m.get("section", ""))}
                for _, d, m in rows]

    return {
        "candidate_id": candidate_id,
        "name": first.get("name", ""),
        "source": first.get("source", ""),
        # Recovered from the header chunk rather than stored, so it works on
        # every CV already in the index. See contact.py for why it only ever
        # looks at the header.
        "phone": phone_from(rows[0][1] if rows else ""),
        # Empty for anything indexed before the pipeline started sending it.
        "source_url": first.get("source_url", ""),
        "extracted_text": True,
        "sections": sections,
    }


@app.post("/index")
def index(candidate: Candidate):
    """Index one CV. This is what the n8n pipeline calls after extraction.

    Idempotent: chunk ids derive from candidate_id, so the same person arriving
    twice updates their entry rather than duplicating it.
    """
    try:
        chunks = index_cv(candidate.candidate_id, candidate.name,
                          candidate.text, candidate.source, candidate.source_url)
    except Exception as exc:
        raise HTTPException(500, f"indexing failed: {type(exc).__name__}: {exc}")
    if chunks == 0:
        raise HTTPException(422, "no indexable text in that CV")

    # Mismatch detector for the caller's item pairing.
    #
    # The n8n workflow builds this request from two different nodes: the email
    # comes from the extraction step, the text from the PDF step. If those get
    # mispaired -- the failure the pipeline's own notes document, where an
    # expression silently resolves to item 0 for every item in a batch -- then
    # every CV in the batch arrives carrying the first candidate's text under
    # its own email. Nothing errors. The index just quietly fills with wrong
    # answers.
    #
    # A CV almost always contains its owner's email address, so a posted email
    # that does not appear in its posted text is a strong signal of exactly that
    # bug. Reported rather than rejected, because a minority of CVs legitimately
    # carry the address only in a header image the extractor cannot read.
    paired = candidate.candidate_id in candidate.text.lower()
    if not paired:
        logger.warning(
            "possible item mispairing: %s not found in the text posted for it "
            "(source=%r). If a whole batch reports this, check the caller's "
            "item pairing before trusting the index.",
            candidate.candidate_id, candidate.source or "unknown")

    return {"candidate_id": candidate.candidate_id, "chunks": chunks,
            "email_found_in_text": paired}


@app.post("/search")
def search_endpoint(brief: Brief):
    try:
        return search(brief.text, brief.k, brief.explain)
    except RuntimeError as exc:
        # Ranking unavailable -- no API key, or the anthropic package missing.
        # Retrieval still works, so degrade to it rather than failing outright.
        #
        # This path must carry `weak` and `top_score` like the normal one. They
        # are what a caller branches on to say "nobody fits" instead of showing
        # five near-misses as a shortlist, and dropping them here made the
        # degraded response quietly less safe than the response it stands in for.
        hits = retrieve(brief.text, brief.k)
        top = hits[0]["score"] if hits else 0.0
        return {"brief": brief.text, "answer": None, "error": str(exc),
                "weak": bool(hits) and top < MIN_SCORE, "top_score": top,
                "matches": [{"candidate_id": h["candidate_id"], "name": h["name"],
                             "section": h["section"], "score": h["score"]}
                            for h in hits]}
    except Exception as exc:
        raise HTTPException(500, f"search failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=store.API_HOST, port=store.API_PORT)
