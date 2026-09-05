"""Sentence embeddings: all-MiniLM-L6-v2 through ONNX Runtime.

Why not sentence-transformers, which is what everyone reaches for:

This service runs on a 3.8 GB VPS that already hosts n8n, a trading dashboard
and a Flask job runner. `pip install sentence-transformers` pulls torch, and the
default torch wheel drags roughly 2.5 GB of CUDA packages onto a machine with no
GPU. ONNX Runtime loads the same model, produces the same 384-dimensional
vectors, and costs about a tenth of the footprint.

The model is the exact one Chroma ships as its default embedding function, from
the same public bucket -- so switching to Chroma's built-in EF, or to
sentence-transformers on a larger box, gives identical vectors and needs no
reindex.

An embedding here is three steps, none of them mysterious:

  1. tokenize      text -> integer token ids
  2. forward pass  ids -> one 384-d vector per *token*
  3. mean pool     average those token vectors, weighted by the attention mask
                   so padding contributes nothing, then L2-normalise

Normalising is what makes cosine similarity and Euclidean distance rank
identically, which is why Chroma's default L2 distance is fine to use directly.
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Sequence

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSIONS = 384
MAX_TOKENS = 256  # a CV section is well under this; longer inputs are truncated

# The same artefact Chroma downloads for its default embedding function.
MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"

CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", Path(__file__).parent / ".model-cache"))


def _ensure_model() -> Path:
    """Download and unpack the model once, into the local cache."""
    onnx_dir = CACHE_DIR / "onnx"
    if (onnx_dir / "model.onnx").exists() and (onnx_dir / "tokenizer.json").exists():
        return onnx_dir

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / "onnx.tar.gz"
    if not archive.exists():
        print(f"[embedder] downloading {MODEL_NAME} (~80 MB, once)...")
        urllib.request.urlretrieve(MODEL_URL, archive)

    with tarfile.open(archive, "r:gz") as tar:
        # Guard against path traversal in the archive rather than trusting it.
        for member in tar.getmembers():
            target = (CACHE_DIR / member.name).resolve()
            if not str(target).startswith(str(CACHE_DIR.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tar.extractall(CACHE_DIR)

    if not (onnx_dir / "model.onnx").exists():
        raise RuntimeError(f"model.onnx not found under {onnx_dir} after extraction")
    return onnx_dir


class Embedder:
    """Loads the model once and holds it. Construction is the expensive part."""

    def __init__(self) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        onnx_dir = _ensure_model()

        self.tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = onnxruntime.SessionOptions()
        # One thread. The box is small and shared; a burst of parallel BLAS
        # threads during indexing is exactly what would disturb the neighbours.
        opts.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "1"))
        opts.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(onnx_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._encode_batch(list(texts[i:i + batch_size])))
        return out

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in self._input_names}

        # last_hidden_state: (batch, tokens, 384)
        hidden = self.session.run(None, feed)[0]

        # Mean pool over real tokens only -- padding must not drag vectors toward
        # the origin, which is what a plain .mean(axis=1) would do.
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-9, None)
        return pooled.astype(np.float32).tolist()


_shared: "Embedder | None" = None


def get_embedder() -> Embedder:
    """Process-wide singleton. The model is ~90 MB resident; load it once."""
    global _shared
    if _shared is None:
        _shared = Embedder()
    return _shared


if __name__ == "__main__":
    emb = get_embedder()

    vecs = emb.encode([
        "settlement reconciliation against issuer statements",  # Ahmed's wording
        "engineer who works on payments APIs",                  # a recruiter's wording
        "registered nurse, intensive care unit",
    ])

    assert len(vecs) == 3 and len(vecs[0]) == DIMENSIONS, [len(v) for v in vecs]
    for v in vecs:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-4, "vectors are not normalised"

    a, b, c = (np.array(v) for v in vecs)
    payments_pair = float(a @ b)   # different words, same meaning
    unrelated_pair = float(a @ c)  # different words, different meaning

    print(f"embedder: {DIMENSIONS}-d, normalised")
    print(f"  settlement/reconciliation  vs  payments APIs : {payments_pair:+.3f}")
    print(f"  settlement/reconciliation  vs  ICU nurse     : {unrelated_pair:+.3f}")

    # This gap is the entire premise of the project: two texts sharing no
    # vocabulary must still land closer than two texts about different jobs.
    assert payments_pair > unrelated_pair + 0.15, (
        f"embeddings do not separate meaning: {payments_pair:.3f} vs {unrelated_pair:.3f}")
    print("embedder: all assertions passed")
