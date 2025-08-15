# pipeline/embed_index.py

import os, json

# Be quiet by default
os.environ.setdefault("CHROMA_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("CHROMA_SERVER_NO_TELEMETRY", "1")

# Optional: modern sqlite on Cloud
try:
    import sys, pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

_EMBEDDER = None
_COLL = None

def get_chroma():
    # Import here (lazy) so our env is set before Chroma loads
    import chromadb
    from chromadb.config import Settings

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

def _get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER

def delete_episode(episode_id):
    coll = get_chroma()
    try:
        coll.delete(where={"episode_id": str(episode_id)})
    except Exception:
        pass

def upsert_episode(chunks, episode_meta, batch_size=200, replace=True):
    if replace and episode_meta.get("episode_id"):
        delete_episode(episode_meta["episode_id"])

    coll = get_chroma()
    emb = _get_embedder()

    ids, docs, metas = [], [], []
    for idx, ch in enumerate(chunks):
        speakers = ch.get("speakers") or {}
        if isinstance(speakers, dict):
            speakers = {str(k): float(v) for k, v in speakers.items()}
        else:
            speakers = {}

        top = max(speakers, key=speakers.get) if speakers else ""
        top_secs = float(speakers[top]) if top else 0.0

        ids.append(f"{episode_meta['episode_id']}_{idx}")
        docs.append(str(ch.get("text", "")))
        metas.append({
            "episode_id": str(episode_meta.get("episode_id", "")),
            "episode_title": str(episode_meta.get("episode_title", "")),
            "start_time": float(ch.get("start", 0.0)),
            "end_time": float(ch.get("end", 0.0)),
            "tokens": int(ch.get("tokens", 0)),
            "n_speakers": int(len(speakers)),
            "top_speaker": str(top),
            "top_speaker_secs": float(top_secs),
            "speakers_json": json.dumps(speakers, ensure_ascii=False),
        })

    if not docs:
        return

    embeddings = emb.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()
    for i in range(0, len(docs), batch_size):
        j = i + batch_size
        coll.upsert(
            ids=ids[i:j],
            documents=docs[i:j],
            metadatas=metas[i:j],
            embeddings=embeddings[i:j],
        )
