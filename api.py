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

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import store
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

    @field_validator("candidate_id")
    @classmethod
    def _looks_like_an_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError("candidate_id must be an email address")
        return v


@app.get("/health")
def health():
    """Liveness plus the one fact worth knowing: is anything actually indexed."""
    try:
        count = store.get_collection().count()
    except Exception as exc:
        raise HTTPException(503, f"vector store unreachable: {type(exc).__name__}")
    return {"status": "ok", "store": store.describe(), "chunks": count,
            "model": store.CLAUDE_MODEL, "min_score": MIN_SCORE}


@app.post("/index")
def index(candidate: Candidate):
    """Index one CV. This is what the n8n pipeline calls after extraction.

    Idempotent: chunk ids derive from candidate_id, so the same person arriving
    twice updates their entry rather than duplicating it.
    """
    try:
        chunks = index_cv(candidate.candidate_id, candidate.name,
                          candidate.text, candidate.source)
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
        # Missing API key: retrieval still works, so degrade to it rather than
        # failing the request outright.
        return {"brief": brief.text, "answer": None, "error": str(exc),
                "matches": [{"candidate_id": h["candidate_id"], "name": h["name"],
                             "section": h["section"], "score": h["score"]}
                            for h in retrieve(brief.text, brief.k)]}
    except Exception as exc:
        raise HTTPException(500, f"search failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=store.API_HOST, port=store.API_PORT)
