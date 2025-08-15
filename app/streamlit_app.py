# =========================
# Global env & shims
# =========================

# Silence Chroma telemetry noise in logs
import os
os.environ.setdefault("CHROMA_TELEMETRY", "0")

# Default to CPU-safe settings (don’t override if user already set these)
os.environ.setdefault("CT2_FORCE_CPU", "1")          # helps faster-whisper on CPU-only runners
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # hide GPUs on shared infra
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Use a modern SQLite for Chroma on Streamlit Cloud / older images
# (Requires: pysqlite3-binary in requirements.txt)
try:
    import sys, pysqlite3  # must run BEFORE any module imports chromadb/sqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

# Robust project-root import shim (so "pipeline/..." and "app/..." can be imported)
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =========================
# Standard imports
# =========================
import streamlit as st
import json as _json
from pathlib import Path

from pipeline.ingest import process_episode
from pipeline.align import assign_speakers, sentences_from_words
from pipeline.chunk import time_aware_windows
from pipeline.embed_index import upsert_episode, get_chroma
from pipeline.retrieve import Retriever
from app.components import ts_to_mmss


# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Podcast RAG", page_icon="🎙️", layout="wide")
st.title("🎙️ Audio-to-Text RAG for Podcast Search")


# =========================
# Helpers (cached)
# =========================
@st.cache_resource(show_spinner=False)
def _get_retriever(rerank: bool):
    """Cache Retriever (downloads models once)."""
    return Retriever(rerank=rerank)

def _safe_json_load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return _json.load(f)


# =========================
# Sidebar: Upload & Index
# =========================
with st.sidebar:
    st.header("Settings")

    # HF token for diarization (optional)
    hf_token_ui = st.text_input(
        "Hugging Face token (for diarization)",
        type="password",
        value=os.getenv("HF_TOKEN", "")
    )
    if hf_token_ui:
        os.environ["HF_TOKEN"] = hf_token_ui

    st.caption("Diarization: " + ("ON (token detected)" if os.getenv("HF_TOKEN") else "OFF (no token)"))

    st.divider()
    st.header("Upload Episodes")
    upl = st.file_uploader("Add audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

    titles = {}
    if upl:
        for f in upl:
            titles[f.name] = st.text_input(f"Title for {f.name}", value=f.name)

    if st.button("Process & Index", type="primary", use_container_width=True):
        if not upl:
            st.error("Upload at least one audio file before indexing.")
            st.stop()

        os.makedirs("storage/data", exist_ok=True)

        with st.spinner("Processing episodes..."):
            ok_count, err_count = 0, 0
            for f in upl:
                try:
                    raw_path = Path("storage/data") / f.name
                    with open(raw_path, "wb") as w:
                        w.write(f.read())

                    ep_json_path, ep_id = process_episode(raw_path, titles[f.name])
                    data = _safe_json_load(Path(ep_json_path))

                    # Assign speakers, build sentences and time-aware windows
                    words = assign_speakers(data["words"], data["turns"])
                    sents = sentences_from_words(words)
                    chunks = time_aware_windows(sents)

                    # Minimal metadata carried into the index
                    meta = {
                        "episode_id": data["episode_id"],
                        "episode_title": data["episode_title"],
                    }
                    upsert_episode(chunks, meta)
                    ok_count += 1
                except Exception as e:
                    err_count += 1
                    st.warning(f"Failed to index {f.name}: {e}")

        if ok_count and not err_count:
            st.success(f"Indexed {ok_count} episode(s). Go search ➡️", icon="✅")
        elif ok_count and err_count:
            st.info(f"Indexed {ok_count} episode(s), {err_count} failed. Check logs.", icon="ℹ️")
        else:
            st.error("No episodes were indexed. Please check the logs.", icon="❌")


st.divider()
st.header("🔎 Search across episodes")

# Determine current index size to cap slider and avoid noisy logs
try:
    _coll = get_chroma()          # collection handle
    _N = int(_coll.count()) if _coll else 0
except Exception:
    _N = 0

colA, colB = st.columns([2, 1], vertical_alignment="bottom")

with colB:
    # Cap at number of chunks in the index (1..15)
    max_k = max(1, min(15, _N or 1))
    k = st.slider("Max results", min_value=1, max_value=max_k, value=min(8, max_k))
    rerank = st.toggle("Re-rank (better precision)", value=False)

with colA:
    query = st.text_input(
        "Ask about topics, quotes, names, steps…",
        placeholder="e.g., What did they say about transfer learning?"
    )

    if st.button("Search", type="primary") and query:
        if _N == 0:
            st.info("No chunks indexed yet. Upload and index episodes in the sidebar.")
        else:
            with st.spinner("Searching…"):
                # Request up to min(15, N) for retrieval, show top-k
                top_for_retrieval = min(15, _N)
                retriever = _get_retriever(rerank=rerank)
                hits = retriever.search(query, k=top_for_retrieval, out_k=k)

            if not hits:
                st.info("No matches found.")
            else:
                for hid, text, meta in hits:
                    with st.container(border=True):
                        # Header line with episode and time span
                        st.markdown(
                            f"**{meta.get('episode_title','(untitled)')}**  ·  "
                            f"⏱️ {ts_to_mmss(meta.get('start_time', 0))} – {ts_to_mmss(meta.get('end_time', 0))}"
                        )
                        # Result text
                        st.write(text)

                        # Speakers (if available)
                        spk_meta = meta.get("speakers_json") or meta.get("speakers")
                        spk_dict = {}
                        if isinstance(spk_meta, str):
                            try:
                                spk_dict = _json.loads(spk_meta)
                            except Exception:
                                spk_dict = {}
                        elif isinstance(spk_meta, dict):
                            spk_dict = spk_meta

                        if spk_dict:
                            st.caption("Speakers: " + ", ".join(map(str, spk_dict.keys())))
                        else:
                            top = meta.get("top_speaker")
                            if top:
                                st.caption(f"Top speaker: {top}")
