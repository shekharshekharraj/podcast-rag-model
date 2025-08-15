# =========================
# Global env & shims
# =========================

# Silence Chroma & misc telemetry noise
import os
os.environ.setdefault("CHROMA_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("CHROMA_SERVER_NO_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Default to CPU-safe settings (don’t override if user already set these)
os.environ.setdefault("CT2_FORCE_CPU", "1")          # helps faster-whisper on CPU-only runners
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # hide GPUs on shared infra
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Optional: allow disabling diarization via secret/var
if os.getenv("ENABLE_DIARIZATION") == "0":
    os.environ.pop("HF_TOKEN", None)

# Use a modern SQLite for Chroma on Streamlit Cloud / older images
# (Requires: pysqlite3-binary in requirements.txt)
try:
    import sys, pysqlite3  # must run BEFORE any module imports chromadb/sqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    import sys  # ensure sys exists even if the try-block above failed
    pass

# Robust project-root import shim (so "pipeline/..." and "app/..." can be imported)
from pathlib import Path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =========================
# Standard imports
# =========================
import streamlit as st
import json as _json

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
        value=os.getenv("HF_TOKEN", ""),
        help="Leave empty to skip diarization (recommended on CPU-run).",
    )
    if hf_token_ui:
        os.environ["HF_TOKEN"] = hf_token_ui

    st.caption("Diarization: " + ("ON (token detected)" if os.getenv("HF_TOKEN") else "OFF (no token)"))

    st.divider()
    st.header("Upload Episodes")
    upl = st.file_uploader("Add audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

    titles = {}
    if upl:
        for i, f in enumerate(upl):
            # unique key avoids widget collisions when filenames repeat
            titles[f.name] = st.text_input(f"Title for {f.name}", value=f.name, key=f"title_{i}_{f.name}")

    # --- INDEX BUTTON (wrapped to surface errors nicely) ---
    if st.button("Process & Index", type="primary", use_container_width=True):
        try:
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

                        ep_json_path, ep_id = process_episode(raw_path, titles.get(f.name, f.name))
                        data = _safe_json_load(Path(ep_json_path))

                        # Assign speakers, build sentences and time-aware windows
                        words = assign_speakers(data["words"], data["turns"])  # turns may be [] if diarization off
                        sents = sentences_from_words(words)
                        chunks = time_aware_windows(sents)

                        # Minimal metadata carried into the index
                        meta = {
                            "episode_id": data["episode_id"],
                            "episode_title": data["episode_title"],
                        }
                        # replace=True ensures re-indexing the SAME episode_id overwrites only its own chunks
                        upsert_episode(chunks, meta, replace=True)
                        ok_count += 1
                    except Exception as ee:
                        err_count += 1
                        st.warning(f"Failed to index {f.name}: {ee}")

            if ok_count and not err_count:
                st.success(f"Indexed {ok_count} episode(s). Go search ➡️", icon="✅")
            elif ok_count and err_count:
                st.info(f"Indexed {ok_count} episode(s), {err_count} failed. Check logs.", icon="ℹ️")
            else:
                st.error("No episodes were indexed. Please check the logs.", icon="❌")
        except Exception as e:
            st.exception(e)


st.divider()
st.header("🔎 Search across episodes")

# Determine current index size to cap slider and avoid noisy logs
try:
    _coll = get_chroma()  # collection handle
    _N = int(_coll.count()) if _coll else 0
except Exception:
    _N = 0

# Precompute episode counts and id->title mapping once (used by Index status and scope picker)
episodes_counts = {}
id_to_title = {}
try:
    if _N > 0:
        _got = _coll.get(include=["metadatas"], limit=5000)
        for m in (_got.get("metadatas") or []):
            if not m:
                continue
            eid = m.get("episode_id")
            if not eid:
                continue
            episodes_counts[eid] = episodes_counts.get(eid, 0) + 1
            if eid not in id_to_title:
                id_to_title[eid] = m.get("episode_title", "(untitled)")
except Exception:
    pass

# --- Debug / Visibility: what's in the index? ---
with st.expander("Index status"):
    try:
        st.json({"total_chunks": _N, "by_episode": episodes_counts})
        # (Optional) wipe the index for a clean slate
        st.markdown("**Danger zone**")
        c1, c2 = st.columns([1, 1])
        with c1:
            confirm = st.checkbox("I understand wiping deletes all chunks")
        with c2:
            if st.button("Wipe index", disabled=not confirm):
                try:
                    _coll.delete(where={})  # delete everything
                    st.success("Index cleared. Re-index episodes.")
                    st.stop()
                except Exception as e:
                    st.exception(e)
    except Exception as e:
        st.caption("(Index status unavailable)")
        st.exception(e)

# =========================
# Search scope + UI
# =========================
colA, colB = st.columns([2, 1], vertical_alignment="bottom")

with colB:
    # Scope: all episodes vs a single episode
    scope = st.radio("Search scope", ["All episodes", "Choose one"], index=0)
    chosen_eid = None
    scope_count = _N

    if scope == "Choose one":
        if episodes_counts:
            # Build stable, informative labels
            options = []
            for eid, cnt in episodes_counts.items():
                title = id_to_title.get(eid, "(untitled)")
                label = f"{title} · {eid[:8]} · {cnt} chunks"
                options.append((label, eid, cnt))
            options.sort(key=lambda x: x[0].lower())

            labels = [o[0] for o in options]
            default_idx = 0
            selected_label = st.selectbox("Episode", labels, index=default_idx)
            # Map label back to (eid, cnt)
            for lbl, eid, cnt in options:
                if lbl == selected_label:
                    chosen_eid = eid
                    scope_count = cnt
                    break
        else:
            st.info("No episodes indexed yet.")
            scope_count = 0

    # Rerank toggle
    rerank = st.toggle("Re-rank (better precision)", value=False)

    # Results slider capped to scope size
    if scope_count <= 1:
        k = 1
        st.caption("Scope has ≤ 1 chunk; showing 1 result.")
    else:
        max_k = min(15, scope_count)
        k = st.slider("Max results", min_value=1, max_value=max_k, value=min(8, max_k))

with colA:
    query = st.text_input(
        "Ask about topics, quotes, names, steps…",
        placeholder="e.g., What did they say about transfer learning?",
    )

    # --- SEARCH BUTTON (wrapped to surface errors nicely) ---
    if st.button("Search", type="primary") and query:
        try:
            if scope_count == 0:
                st.info("No chunks in the current scope. Upload & index episodes first.")
            else:
                with st.spinner("Searching…"):
                    # Prepare filters if a single episode is chosen
                    filters = {"episode_id": chosen_eid} if chosen_eid else None
                    top_for_retrieval = min(15, scope_count)
                    retriever = _get_retriever(rerank=rerank)
                    hits = retriever.search(query, k=top_for_retrieval, out_k=k, filters=filters)

                if not hits:
                    st.info("No matches found for the current scope.")
                else:
                    for hid, text, meta in hits:
                        with st.container(border=True):
                            st.markdown(
                                f"**{meta.get('episode_title','(untitled)')}**  ·  "
                                f"⏱️ {ts_to_mmss(meta.get('start_time', 0))} – "
                                f"{ts_to_mmss(meta.get('end_time', 0))}"
                            )
                            st.write(text)

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
        except Exception as e:
            st.exception(e)
