# pipeline/embed_index.py

# --- Quiet logs & ensure modern sqlite for Chroma (works on Streamlit Cloud) ---
import os
os.environ.setdefault("CHROMA_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("CHROMA_SERVER_NO_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import sys, pysqlite3  # provided by pysqlite3-binary in requirements.txt
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import json
import chromadb
from chromadb.config import Settings

# (last-resort) silence telemetry capture if still present
try:
    import chromadb.telemetry.telemetry as _ctel
    if hasattr(_ctel, "posthog") and hasattr(_ctel.posthog, "capture"):
        _ctel.posthog.capture = lambda *a, **k: None
except Exception:
    pass

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


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy import to avoid pulling torch at app boot."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def delete_episode(episode_id: str | int):
    """Remove all chunks for an episode before re-inserting (prevents duplicate-ID errors)."""
    coll = get_chroma()
    try:
        coll.delete(where={"episode_id": str(episode_id)})
    except Exception:
        # Safe to ignore if nothing to delete
        pass


def upsert_episode(chunks, episode_meta, batch_size: int = 200, replace: bool = True):
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

    If replace=True, we first delete existing vectors for this episode_id.
    """
    episode_id = str(episode_meta.get("episode_id", ""))
    if replace and episode_id:
        delete_episode(episode_id)

    coll = get_chroma()
    emb = _get_embedder()

    ids, docs, metas = [], [], []

    for idx, ch in enumerate(chunks):
        text = str(ch.get("text", ""))
        start = float(ch.get("start", 0.0))
        end = float(ch.get("end", start))
        tokens = int(ch.get("tokens", 0))

        speakers_raw = ch.get("speakers") or {}
        if isinstance(speakers_raw, dict):
            speakers_clean = {str(k): float(v) for k, v in speakers_raw.items()}
        else:
            speakers_clean = {}

        if speakers_clean:
            top_speaker = max(speakers_clean, key=speakers_clean.get)
            top_speaker_secs = float(speakers_clean[top_speaker])
        else:
            top_speaker, top_speaker_secs = "", 0.0

        ids.append(f"{episode_id}_{idx}")
        docs.append(text)
        metas.append({
            "episode_id": episode_id,
            "episode_title": str(episode_meta.get("episode_title", "")),
            "start_time": start,
            "end_time": end,
            "tokens": tokens,
            "n_speakers": int(len(speakers_clean)),
            "top_speaker": str(top_speaker),
            "top_speaker_secs": float(top_speaker_secs),
            "speakers_json": json.dumps(speakers_clean, ensure_ascii=False),
        })

    if not docs:
        return

    # Embed once for all docs, then upsert in batches to avoid memory spikes
    embeddings = _get_embedder().encode(
        docs, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    for i in range(0, len(docs), batch_size):
        j = i + batch_size
        coll.upsert(
            ids=ids[i:j],
            documents=docs[i:j],
            metadatas=metas[i:j],
            embeddings=embeddings[i:j],
        )
