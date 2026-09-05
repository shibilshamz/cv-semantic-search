"""Chunk, embed and store CVs.

Called two ways:

  from the CLI   python indexer.py --corpus corpus/synthetic
  from the API   POST /index, which is what the n8n CV pipeline calls once it
                 has extracted a candidate's text

candidate_id is the candidate's email address -- already the deduplication key
in the n8n workflow, so the two systems agree on identity without a mapping
table. Chunk ids are f"{candidate_id}:{i}", which makes re-indexing the same CV
an upsert over the same rows rather than a second copy of the person. A CV
arriving twice (re-uploaded, or sent by two agencies) therefore costs nothing.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from chunker import split_sections
from embedder import get_embedder
from store import get_collection

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def index_cv(candidate_id: str, name: str, text: str, source: str = "") -> int:
    """Index one CV. Returns the number of chunks written."""
    chunks = split_sections(text)
    if not chunks:
        return 0

    col = get_collection()
    vectors = get_embedder().encode([c.text for c in chunks])

    col.upsert(
        ids=[f"{candidate_id}:{i}" for i in range(len(chunks))],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        # The metadata is what lets retrieval collapse many chunks back into one
        # candidate, and what gives the recruiter a reason for the match.
        metadatas=[{
            "candidate_id": candidate_id,
            "name": name,
            "section": c.section,
            "source": source,
        } for c in chunks],
    )
    _prune(col, candidate_id, len(chunks))
    return len(chunks)


def _prune(col, candidate_id: str, keep: int) -> None:
    """Drop stale chunks from a previous, longer version of the same CV.

    Upsert overwrites ids 0..keep-1 but leaves 0..n-1 from an earlier revision in
    place. Without this, editing a CV down from seven sections to four would
    leave three orphan chunks that still answer queries with text the candidate
    has removed.
    """
    stale = [f"{candidate_id}:{i}" for i in range(keep, keep + 12)]
    existing = col.get(ids=stale, include=[])
    if existing and existing.get("ids"):
        col.delete(ids=existing["ids"])


def index_corpus(directory: Path) -> Dict[str, int]:
    """Index a directory of .txt CVs, using manifest.json for identity if present."""
    manifest_path = directory / "manifest.json"
    records: List[dict] = []

    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        # No manifest: fall back to scraping the first email out of each file.
        # The live path never uses this -- n8n supplies the email from the Claude
        # extraction -- but it keeps the CLI usable on an ad-hoc folder.
        for f in sorted(directory.glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            found = EMAIL_RE.search(text)
            if not found:
                print(f"  skip {f.name}: no email, no candidate id")
                continue
            records.append({"file": f.name, "candidate_id": found.group(0).lower(),
                            "name": text.strip().split("\n")[0][:60]})

    chunks = 0
    for r in records:
        path = directory / r["file"]
        if not path.exists():
            print(f"  skip {r['file']}: missing")
            continue
        chunks += index_cv(
            candidate_id=r["candidate_id"].lower(),
            name=r["name"],
            text=path.read_text(encoding="utf-8"),
            source=r["file"],
        )

    return {"files": len(records),
            "candidates": len({r["candidate_id"].lower() for r in records}),
            "chunks": chunks}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Index CVs into the vector store.")
    ap.add_argument("--corpus", type=Path, default=Path("corpus/synthetic"),
                    help="directory of .txt CVs (default: corpus/synthetic)")
    ap.add_argument("--reset", action="store_true",
                    help="delete the collection first, for a clean rebuild")
    args = ap.parse_args(argv)

    if args.reset:
        from store import COLLECTION, get_client
        try:
            get_client().delete_collection(COLLECTION)
            print(f"deleted collection {COLLECTION!r}")
        except Exception as exc:
            print(f"nothing to delete ({type(exc).__name__})")

    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")

    print(f"indexing {args.corpus} ...")
    stats = index_corpus(args.corpus)
    total = get_collection().count()
    print(f"  {stats['files']} files, {stats['candidates']} candidates, "
          f"{stats['chunks']} chunks written")
    print(f"  collection now holds {total} chunks")

    # More chunks than candidates is the whole design. If these were equal the
    # chunker had silently stopped splitting, and retrieval quality would drop
    # without anything failing.
    if stats["candidates"] and stats["chunks"] <= stats["candidates"]:
        raise SystemExit("chunks <= candidates: section splitting is not working")


if __name__ == "__main__":
    main()
