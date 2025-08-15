# pipeline/embed_index.py

# --- Quiet logs & ensure modern sqlite for Chroma (works on Streamlit Cloud) ---
import os
os.environ.setdefault("CHROMA_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("CHROMA_SERVER_NO_TELEMETRY", "1")

try:
    import sys, pysqlite3  # provided by pysqlite3-binary in requirements.txt
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    # If not available locally, Chroma will try normal sqlite; that's fine on dev
    pass

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# (last-resort) hard-mute telemetry capture if a plugin still tries to emit
try:
    import chromadb.telemetry.telemetry as _ctel
    if hasattr(_ctel, "posthog") and hasattr(_ctel.posthog, "capture"):
        _ctel.posthog.capture = lambda *a, **k: None
except Exception:
    pass

# --- Module-level singletons to avoid repeated downloads/initializations ---
_EMBEDDER = None
_COLL = None


def get_chroma():
    """Return a persistent Chroma collection with telemetry disabled."""
    global _COLL
    if _COLL is None:
        client = chromadb.PersistentClient(
            path="storage/chroma",
            settings=Settings(anonymized_telemetry=False),
        )
        _COLL = client.get_or_create_collection(
            name="podcast_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _COLL


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Reuse HF cache; show_progress_bar off for cleaner logs
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(model_name)
    return _EMBEDDER


def upsert_episode(chunks, episode_meta):
    """
    Upsert a list of chunk dicts:
      {
        "text": str,
        "start": float,
        "end": float,
        "tokens": int,
        "speakers": { "SPK0": seconds_float, ... }  # optional
      }
    Episode metadata must have: {"episode_id": str/int, "episode_title": str}
    """
    coll = get_chroma()
    emb = _get_embedder()

    ids, docs, metas = [], [], []

    for idx, ch in enumerate(chunks):
        # Safe field extraction with sane defaults
        text = str(ch.get("text", ""))
        start = float(ch.get("start", 0.0))
        end = float(ch.get("end", start))
        tokens = int(ch.get("tokens", 0))

        # Speakers dict -> ensure {str: float}; keep full map as JSON string only
        speakers_raw = ch.get("speakers") or {}
        if isinstance(speakers_raw, dict):
            speakers_clean = {str(k): float(v) for k, v in speakers_raw.items()}
        else:
            speakers_clean = {}

        # Primitive-only summary fields for Chroma metadata
        if speakers_clean:
            top_speaker = max(speakers_clean, key=speakers_clean.get)
            top_speaker_secs = float(speakers_clean[top_speaker])
        else:
            top_speaker, top_speaker_secs = "", 0.0

        ids.append(f"{episode_meta['episode_id']}_{idx}")
        docs.append(text)
        metas.append(
            {
                # episode info
                "episode_id": str(episode_meta.get("episode_id", "")),
                "episode_title": str(episode_meta.get("episode_title", "")),

                # time window
                "start_time": start,
                "end_time": end,

                # token stats
                "tokens": tokens,

                # speaker summaries (primitive types only)
                "n_speakers": int(len(speakers_clean)),
                "top_speaker": str(top_speaker),
                "top_speaker_secs": float(top_speaker_secs),

                # full map as JSON string (NOT a dict)
                "speakers_json": json.dumps(speakers_clean, ensure_ascii=False),
            }
        )

    if docs:
        embeddings = emb.encode(docs)
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
