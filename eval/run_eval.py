"""Measure retrieval quality against eval/briefs.yaml.

    python eval/run_eval.py

Retrieval only -- no Claude call, so it is free and deterministic, and a
regression points at the chunker or the embedder rather than at model variance.

Two metrics:

  recall@k  did the expected candidate appear in the top k at all
  MRR       mean reciprocal rank -- 1.0 if first, 0.5 if second, and so on.
            Recall alone hides the difference between "ranked first" and
            "scraped in at position five", which is exactly the difference a
            recruiter notices.

Plus a negative control: briefs with no expected answer must produce a top score
below the threshold, so the service can say "no strong match" instead of
returning five people who happen to be nearest in a corpus that contains nobody
suitable.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import retrieve  # noqa: E402
from store import describe  # noqa: E402

SPEC = Path(__file__).parent / "briefs.yaml"


def main() -> int:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    k = spec.get("k", 5)
    threshold = spec.get("threshold", 0.35)
    briefs = spec["briefs"]

    print(f"store: {describe()}   k={k}   threshold={threshold}\n")

    positives, hits_at_k, rr_total = 0, 0, 0.0
    negatives, negatives_ok = 0, 0
    failures = []

    for b in briefs:
        results = retrieve(b["brief"], k)
        ids = [r["candidate_id"] for r in results]
        top = results[0]["score"] if results else 0.0
        expect = b.get("expect") or []

        if expect:
            positives += 1
            rank = next((i + 1 for i, cid in enumerate(ids) if cid in expect), None)
            if rank:
                hits_at_k += 1
                rr_total += 1.0 / rank
                mark, detail = "PASS", f"rank {rank}, score {top:+.3f}"
            else:
                mark, detail = "FAIL", f"expected {expect[0]} - not in top {k}"
                failures.append(b["id"])
        else:
            negatives += 1
            if top < threshold:
                negatives_ok += 1
                mark, detail = "PASS", f"top score {top:+.3f} < {threshold} (correctly weak)"
            else:
                mark, detail = "FAIL", f"top score {top:+.3f} >= {threshold} (false confidence)"
                failures.append(b["id"])

        print(f"  [{mark}] {b['id']:<24} {detail}")
        if b.get("note"):
            print(f"         {b['note']}")

    recall = hits_at_k / positives if positives else 0.0
    mrr = rr_total / positives if positives else 0.0

    print(f"\n  recall@{k}          {recall:.2f}  ({hits_at_k}/{positives})")
    print(f"  MRR                {mrr:.2f}")
    print(f"  negative control   {negatives_ok}/{negatives}")

    # Separation is the number that actually predicts whether the threshold will
    # hold on a corpus this eval has never seen.
    pos_tops = [retrieve(b["brief"], k)[0]["score"]
                for b in briefs if b.get("expect")]
    neg_tops = [retrieve(b["brief"], k)[0]["score"]
                for b in briefs if not b.get("expect")]
    if pos_tops and neg_tops:
        print(f"  separation         worst genuine {min(pos_tops):+.3f} vs "
              f"best non-match {max(neg_tops):+.3f}")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nall briefs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
