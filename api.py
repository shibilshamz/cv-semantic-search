"""HTTP interface.

    uvicorn api:app --host $API_HOST --port $API_PORT

Bound by default to 172.17.0.1 -- the docker0 bridge address -- not 0.0.0.0 and
not 127.0.0.1:

  127.0.0.1  invisible to the n8n container, which is the one client that needs
             to reach POST /index
  0.0.0.0    reachable from the internet unless ufw says otherwise, which makes
             the firewall the only thing standing between a candidate database
             and the public. That is how job-runner on 5679 is set up, and it is
             the part of that setup worth not copying.
  172.17.0.1 reachable from containers on the default bridge, and from nowhere
             else, by virtue of the address itself rather than a rule that
             someone might change

Set API_HOST=127.0.0.1 in .env for local development.
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import store
from indexer import index_cv
from search import MIN_SCORE, retrieve, search

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
    candidate_id: str = Field(..., description="the candidate's email; the dedup key")
    name: str
    text: str = Field(..., min_length=1, description="extracted CV text")
    source: str = ""


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
        chunks = index_cv(candidate.candidate_id.strip().lower(), candidate.name,
                          candidate.text, candidate.source)
    except Exception as exc:
        raise HTTPException(500, f"indexing failed: {type(exc).__name__}: {exc}")
    if chunks == 0:
        raise HTTPException(422, "no indexable text in that CV")
    return {"candidate_id": candidate.candidate_id.strip().lower(), "chunks": chunks}


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
