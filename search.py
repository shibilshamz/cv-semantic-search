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
from typing import Dict, List

from embedder import get_embedder
from store import CLAUDE_MODEL, get_collection, similarity

# The grounding instruction. This is the load-bearing line in the file.
#
# Without it the model fills gaps from its own priors and describes experience
# the candidate does not have -- confidently, in fluent prose, attributed to a
# real person who is about to be put in front of a client. Everything else here
# is plumbing; this is the part that makes the output safe to act on.
SYSTEM = (
    "You rank candidates against a job brief. Use ONLY the excerpts provided. "
    "Cite the [candidate_id] behind every claim. If an excerpt does not support "
    "a requirement, say so explicitly. Never infer experience that is not "
    "written down. If none of the candidates is a good fit, say that plainly "
    "rather than ranking weak matches as though they were strong."
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


def rank(brief: str, results: List[Dict]) -> str:
    """Ask Claude to rank the retrieved candidates. Requires ANTHROPIC_API_KEY."""
    if not results:
        return "No candidates are indexed, so there is nothing to rank."

    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; retrieval works, ranking does not")

    context = "\n\n".join(
        f"[{r['candidate_id']}] {r['name']} - {r['section']}\n{r['text']}"
        for r in results
    )
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
