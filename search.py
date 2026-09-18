"""Retrieve candidates for a brief, then have Claude rank them.

Two stages, and it matters that they are separate:

  retrieve()  cheap, deterministic, no model call. Narrows 200 chunks to 5
              candidates by vector distance.
  rank()      one Claude call over those 5. Never sees the other 195.

That split is the point of retrieval-augmented generation. Sending 500 CVs to a
model per query would be slow and expensive; sending 5 costs a fraction of a
cent and takes about a second.
"""

import os
from typing import Dict, List, Optional

from embedder import get_embedder
from store import CLAUDE_MODEL, get_collection, similarity

# The grounding instruction. This is the load-bearing line in the file.
#
# Without it the model fills gaps from its own priors and describes experience
# the candidate does not have -- confidently, in fluent prose, attributed to a
# real person who is about to be put in front of a client. Everything else here
# is plumbing; this is the part that makes the output safe to act on.
SYSTEM = (
    "You rank candidates against a job brief. Use ONLY the CV text provided. "
    "Cite the [candidate_id] behind every claim. Never infer experience that is "
    "not written down. If none of the candidates is a good fit, say that plainly "
    "rather than ranking weak matches as though they were strong.\n\n"
    # Saying a CV lacks something is a claim like any other, and has to be
    # checked like one. The ranker used to see a single section per candidate
    # and reported what that section lacked as what the CV lacked -- describing
    # the one qualified nurse in the index as missing ventilator experience her
    # CV states outright. It now receives every section, so this can be strict.
    "Before writing that a candidate lacks a requirement, read every section of "
    "their CV above. Only then may you call it missing, and say it as \"not "
    "stated in the CV text provided\". If any section mentions it, the "
    "requirement is met: say so and quote the words. Never report something as "
    "absent because the matched section alone did not mention it.\n\n"
    # Added after an early run inferred a candidate's location from their name
    # ("Name and context suggest Gulf region"). In a hiring tool that is both a
    # grounding failure and a discrimination risk, so it is called out by name
    # rather than left to the general instruction above.
    "Never infer nationality, location, ethnicity, gender, age or religion from "
    "a candidate's name. If the brief asks about location, residency or work "
    "authorisation and no section states it, answer \"not stated in the CV text "
    "provided\" and treat the requirement as unmet."
)


def retrieve(brief: str, k: int = 5) -> List[Dict]:
    """Return the k best candidates, one entry each.

    Over-fetches because chunking is section-level: a single strong candidate can
    legitimately occupy five of the top ten slots with five different sections of
    their own CV. Collapsing on candidate_id afterwards -- keeping each person's
    single best-scoring chunk -- turns a list of passages back into a list of
    people, which is what the recruiter asked for.
    """
    if not brief.strip():
        return []

    col = get_collection()
    total = col.count()
    if total == 0:
        return []

    hits = col.query(
        query_embeddings=get_embedder().encode([brief]),
        n_results=min(k * 4, total),
        include=["documents", "metadatas", "distances"],
    )

    best: Dict[str, Dict] = {}
    for doc, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        cid = meta["candidate_id"]
        if cid not in best or dist < best[cid]["distance"]:
            best[cid] = {
                "candidate_id": cid,
                "name": meta.get("name", ""),
                "section": meta.get("section", ""),
                "source": meta.get("source", ""),
                "text": doc,
                "distance": dist,
                "score": similarity(dist),
            }

    return sorted(best.values(), key=lambda r: r["distance"])[:k]


HEADER_CHARS = 400


def _sections_for(candidate_ids: List[str]) -> Dict[str, List[tuple]]:
    """Every indexed section of each candidate's CV, in document order.

    Retrieval hands the ranker one chunk per person -- the best-scoring one --
    and that is the right unit for *finding* someone and the wrong unit for
    *judging* them. A brief asking for ventilator experience matched a nurse on
    her Professional Summary, which does not mention ventilators; the role three
    chunks later says "Ventilator management". The model was shown the summary
    alone, correctly reported that it did not see the skill, and phrased it as
    "not stated in her CV". The only qualified candidate in the index was
    described as unqualified, in the most authoritative-looking block on the
    page, and nothing on screen contradicted it.

    That is the inverse of the failure the system prompt was written to prevent.
    It guards against claiming experience nobody has; a claimed *absence* is the
    one that gets believed, because nobody thinks to check a negative.

    So the ranker now sees the whole CV of everyone it ranks. One batched read
    for the whole result set, not one per candidate. At k=5 this is a few
    thousand extra tokens into a 200k window -- far cheaper than the outcome it
    prevents.
    """
    if not candidate_ids:
        return {}
    try:
        got = get_collection().get(
            where={"candidate_id": {"$in": list(candidate_ids)}},
            include=["documents", "metadatas"],
        )
    except Exception:
        return {}

    by_candidate: Dict[str, List[tuple]] = {}
    for chunk_id, doc, meta in zip(got.get("ids") or [],
                                   got.get("documents") or [],
                                   got.get("metadatas") or []):
        by_candidate.setdefault(meta.get("candidate_id", ""), []).append(
            (chunk_id, meta.get("section", ""), doc))

    # Order by the integer suffix of the chunk id. Sorting the id strings would
    # put ":10" before ":2" and hand the model a shuffled CV.
    def index_of(row):
        tail = row[0].rsplit(":", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    for rows in by_candidate.values():
        rows.sort(key=index_of)
    return by_candidate


def _context_for(r: Dict, sections: Optional[List[tuple]] = None) -> str:
    """One candidate's block for the ranking prompt.

    The section retrieval actually matched on is marked, so the model can still
    tell what the search fired on; the rest is there so it can check a claim
    before making it.
    """
    block = f"[{r['candidate_id']}] {r['name']}"

    if not sections:
        # No batched read (an empty index, or the store errored). Fall back to
        # the matched chunk alone rather than failing the ranking -- but say so,
        # so the model does not read a partial CV as a complete one.
        header = _header_for(r["candidate_id"], r["text"])
        if header:
            block += f"\n-- CV header --\n{header}"
        block += "\n-- NOTE: only the best-matching section was available --"
        return f"{block}\n-- {r['section']} --\n{r['text']}"

    parts = [block, "-- full CV text, every section --"]
    for _, section, text in sections:
        marker = "  <-- the section this brief matched" if section == r["section"] else ""
        parts.append(f"-- {section or 'Section'} --{marker}\n{text}")
    return "\n".join(parts)


def _header_for(candidate_id: str, best_text: str) -> str:
    """The candidate's contact block -- chunk 0 -- if it is not already the match.

    Only used by the degraded path in _context_for now that the ranker normally
    receives every section. Kept because that path still needs location, visa
    status and nationality, which live in the header: without them the model was
    once asked "is this person in the Gulf?" while holding text that never said,
    and answered by guessing from the candidate's name.
    """
    try:
        got = get_collection().get(ids=[f"{candidate_id}:0"], include=["documents"])
        docs = got.get("documents") or []
    except Exception:
        return ""
    if not docs or not docs[0] or docs[0] == best_text:
        return ""
    return docs[0][:HEADER_CHARS]


def rank(brief: str, results: List[Dict]) -> str:
    """Ask Claude to rank the retrieved candidates. Requires ANTHROPIC_API_KEY."""
    if not results:
        return "No candidates are indexed, so there is nothing to rank."

    # Both failures below are the same thing to a caller -- ranking is
    # unavailable, retrieval still works -- so both raise RuntimeError, which is
    # what api.py catches to degrade to matches-only instead of 500ing and
    # throwing the retrieved candidates away with it.
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(f"anthropic is not installed; retrieval works, ranking does not ({exc})")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; retrieval works, ranking does not")

    sections = _sections_for([r["candidate_id"] for r in results])
    context = "\n\n".join(
        _context_for(r, sections.get(r["candidate_id"])) for r in results)
    msg = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Brief:\n{brief}\n\nCandidates:\n{context}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


# Below this cosine score, the best match is not really a match. Calibrated from
# eval/run_eval.py, which reports the separation directly: on the synthetic
# corpus the worst genuine top match scores +0.413 and the best non-match +0.307.
# 0.35 sits in that gap. Re-check it with the eval after any change to chunking
# or the embedding model -- it is a property of this corpus, not a universal.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))


def search(brief: str, k: int = 5, explain: bool = True) -> Dict:
    """Full query path. explain=False skips the Claude call (and its cost)."""
    hits = retrieve(brief, k)
    top = hits[0]["score"] if hits else 0.0
    return {
        "brief": brief,
        # A caller that ignores everything else should still be able to see that
        # nobody in the corpus fits. The ranking prompt says the same thing in
        # prose; this says it in a field you can branch on.
        "weak": bool(hits) and top < MIN_SCORE,
        "top_score": top,
        "answer": rank(brief, hits) if (explain and hits) else None,
        "matches": [{
            "candidate_id": h["candidate_id"],
            "name": h["name"],
            "section": h["section"],
            "score": h["score"],
        } for h in hits],
    }


if __name__ == "__main__":
    import sys

    brief = " ".join(sys.argv[1:]) or (
        "backend engineer comfortable with payments APIs, based in the Gulf")
    hits = retrieve(brief)
    if not hits:
        raise SystemExit("nothing indexed -- run: python indexer.py --corpus corpus/synthetic")

    print(f"brief: {brief}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h['name']:<26} {h['score']:+.3f}  [{h['section']}]")
