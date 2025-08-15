# pipeline/retrieve.py

# --- Quiet logs & (belt-and-suspenders) modern sqlite for Chroma ---
import os
os.environ.setdefault("CHROMA_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("CHROMA_SERVER_NO_TELEMETRY", "1")

try:
    import sys, pysqlite3  # provided by pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

from sentence_transformers import SentenceTransformer, CrossEncoder
from .embed_index import get_chroma  # reuse the same persistent client/collection


# --- Model caches (avoid repeated downloads on serverless runners) ---
_EMBEDDER = None
_CROSS = None


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def _get_cross(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    global _CROSS
    if _CROSS is None:
        _CROSS = CrossEncoder(model_name)
    return _CROSS


class Retriever:
    """
    Vector retriever with optional CrossEncoder reranking.

    Args:
        embed_model: sentence-transformers model used for query embeddings.
        rerank: if True, apply CrossEncoder reranking after vector search.
        cross_model: cross-encoder model id.
    """
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank: bool = False,
        cross_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = _get_embedder(embed_model)
        self.coll = get_chroma()
        self.rerank = rerank
        self.cross_model_name = cross_model

    def search(self, query: str, k: int = 15, out_k: int = 6, filters: dict | None = None):
        """
        Run semantic search with optional reranking.

        Returns:
            List[Tuple[id, document, metadata]]
        """
        # Clamp to index size to avoid warnings/errors
        try:
            total = int(self.coll.count())
        except Exception:
            total = 0

        if total <= 0:
            return []

        k = max(1, min(k, total))
        out_k = max(1, min(out_k, k))

        # Embed query and retrieve
        qv = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()
        res = self.coll.query(query_embeddings=qv, n_results=k, where=filters or {})

        ids = res.get("ids", [[]])[0] if res else []
        docs = res.get("documents", [[]])[0] if res else []
        metas = res.get("metadatas", [[]])[0] if res else []

        hits = list(zip(ids, docs, metas))

        # Optional reranking
        if self.rerank and hits:
            cross = _get_cross(self.cross_model_name)
            pairs = [[query, d] for _, d, _ in hits]
            scores = cross.predict(pairs).tolist()
            # sort by score desc
            hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]

        return hits[:out_k]
