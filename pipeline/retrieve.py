# pipeline/retrieve.py
import os
os.environ.setdefault("CHROMA_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("CHROMA_SERVER_NO_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# modern sqlite for Chroma
try:
    import sys, pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

from .embed_index import get_chroma

_EMBEDDER = None
_CROSS = None

def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER

def _get_cross(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _CROSS
    if _CROSS is None:
        from sentence_transformers import CrossEncoder
        _CROSS = CrossEncoder(model_name)
    return _CROSS

class Retriever:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank: bool = False,
        cross_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embed_model = embed_model
        self.rerank = rerank
        self.cross_model = cross_model
        self.coll = get_chroma()

    def search(self, query: str, k: int = 15, out_k: int = 6, filters: dict | None = None):
        try:
            total = int(self.coll.count())
        except Exception:
            total = 0
        if total <= 0:
            return []

        k = max(1, min(k, total))
        out_k = max(1, min(out_k, k))

        embedder = _get_embedder(self.embed_model)
        qv = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()

        res = self.coll.query(query_embeddings=qv, n_results=k, where=filters or {})
        ids = res.get("ids", [[]])[0] if res else []
        docs = res.get("documents", [[]])[0] if res else []
        metas = res.get("metadatas", [[]])[0] if res else []
        hits = list(zip(ids, docs, metas))

        if self.rerank and hits:
            cross = _get_cross(self.cross_model)
            pairs = [[query, d] for _, d, _ in hits]
            scores = cross.predict(pairs).tolist()
            hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]

        return hits[:out_k]
