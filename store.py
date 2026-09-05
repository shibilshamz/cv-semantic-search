"""Configuration and the Chroma collection handle.

Two modes, chosen by environment:

  CHROMA_PATH set    a local PersistentClient, no Docker needed. This is what
                     makes the repo runnable by anyone who clones it, and what
                     eval/run_eval.py uses.
  otherwise          an HttpClient against the container from docker-compose.yml,
                     which is how it runs on the VPS.

Embeddings are always computed in *this* process (see embedder.py), never by
Chroma, so the collection is created with embedding_function=None and every
write passes vectors explicitly. That keeps one model, one tokenizer and one
version of the truth about what a vector means.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_PATH = os.getenv("CHROMA_PATH")  # set = local mode
COLLECTION = os.getenv("CHROMA_COLLECTION", "cvs")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "5680"))

# Pinned rather than the bare `claude-haiku-4-5` alias: the point of this repo is
# a reproducible eval, and an alias that moves under you invalidates the numbers
# in the README. Same pin as scorer.py and the CV extraction workflow.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")


def get_client():
    import chromadb

    if CHROMA_PATH:
        return chromadb.PersistentClient(path=CHROMA_PATH)
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_collection():
    return get_client().get_or_create_collection(COLLECTION, embedding_function=None)


def describe() -> str:
    return f"chroma://{CHROMA_PATH}" if CHROMA_PATH else f"http://{CHROMA_HOST}:{CHROMA_PORT}"


def similarity(distance: float) -> float:
    """Chroma returns squared L2. Vectors are L2-normalised, so ||a-b||^2 =
    2 - 2*cos(a,b), and cosine similarity comes back exactly."""
    return round(1.0 - (distance / 2.0), 4)
