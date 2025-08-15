🎙️ Audio-to-Text RAG for Podcast Search

Upload podcast episodes → transcribe (word timestamps) → (optional) diarize speakers → time-aware chunking → vector search across episodes → query with timestamped results in a clean Streamlit UI.

Built for local dev (Chroma, CPU-friendly), easy GPU upgrade, and one-click deploy to Hugging Face Spaces.

<img width="1918" height="965" alt="rag1" src="https://github.com/user-attachments/assets/483b8170-b224-4279-a52b-019fbbbe4dc4" />
<img width="1317" height="841" alt="rag2" src="https://github.com/user-attachments/assets/bb3fec99-d860-4022-8005-893341d865fd" />
<img width="417" height="850" alt="rag3" src="https://github.com/user-attachments/assets/631c6323-689e-4798-8273-37af88f50d5a" />

Built for local dev (Chroma, CPU-friendly), easy GPU upgrade, and one-click deploy to Hugging Face Spaces.


Table of Contents

* [Features](#features)
* [Architecture](#architecture)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Quickstart](#quickstart)

  * [Windows (PowerShell, Python 3.11, CPU)](#windows-powershell-python-311-cpu)
  * [macOS / Linux](#macos--linux)
* [Environment Variables](#environment-variables)
* [Using the App](#using-the-app)
* [Deployment](#deployment)

  * [Hugging Face Spaces](#hugging-face-spaces)
  * [Docker](#docker)
* [Troubleshooting](#troubleshooting)
* [Security & Privacy](#security--privacy)
* [License & Credits](#license--credits)

## Features

* Whisper (**faster-whisper**) with **word-level timestamps**
* **Optional** speaker diarization via **pyannote.audio**
* Time-aware chunking (with overlap) to preserve context
* Vector search with **Chroma** (local, persistent)
* Optional **CrossEncoder** reranking for tighter precision
* Streamlit UI: episode title, `MM:SS` range, speaker hints
* Runs well on **CPU**; **GPU** is plug-and-play later

flowchart TD
    A[Audio (.mp3/.wav/.m4a)] --> B[FFmpeg resample → 16k mono WAV]
    B --> C[Whisper (faster-whisper)\nwords + timestamps]
    C -->|optional| D[pyannote.audio diarization\nwho spoke when]
    C --> E[Align words ↔ speakers → sentences]
    E --> F[Time-aware windows\n(~400 tokens, ~20% overlap)]
    F --> G[Embeddings (Sentence-Transformers)]
    G --> H[Chroma (persistent store)]
    I[Query] --> J[kNN retrieve ± CrossEncoder rerank]
    H --> J --> K[Streamlit UI\nepisode + timestamped snippets]
```


## Repository Structure

```
podcast-rag/
├─ app/
│  ├─ streamlit_app.py      # UI + search
│  ├─ components.py         # small UI helpers
│  └─ prompts.py            # (optional) LLM templating
├─ pipeline/
│  ├─ ingest.py             # resample → whisper → (optional) diarize → JSON
│  ├─ align.py              # assign speakers to words + sentence building
│  ├─ chunk.py              # time-aware windowing with overlap
│  ├─ embed_index.py        # embeddings + Chroma upsert
│  └─ retrieve.py           # retrieval + optional rerank
├─ storage/
│  ├─ chroma/               # vector DB (gitignored)
│  └─ data/                 # uploaded audio + episode JSON
├─ eval/
│  └─ dataset.jsonl         # (optional) RAG evaluation sets
├─ requirements.txt
├─ Dockerfile
└─ README.md
```



## Requirements

* **Python 3.11** (recommended on Windows for stable wheels)
* **FFmpeg** CLI (or rely on bundled `imageio-ffmpeg` path in code)
* CPU works fine; GPU requires a matching CUDA/cuDNN stack



## Quickstart

### Windows (PowerShell, Python 3.11, CPU)

```powershell
cd D:\podcast-rag

# Create & activate venv (allow scripts if prompted)
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate

# Install deps
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

# CPU-friendly flags (avoid CUDA DLL issues)
$Env:CT2_FORCE_CPU = "1"
$Env:CUDA_VISIBLE_DEVICES = ""
$Env:WHISPER_MODEL = "medium"   # use "large-v3" later for best accuracy

# (Optional) enable diarization
# $Env:HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"

# Launch
python -m streamlit run app\streamlit_app.py
```

Quick test WAV (PowerShell TTS):**

```powershell
Add-Type -AssemblyName System.Speech
$tts = New-Object System.Speech.Synthesis.SpeechSynthesizer
$tts.SetOutputToWaveFile("episode1.wav")
$tts.Speak("This is a quick test of our podcast RAG system. We mention transfer learning, Chroma, and diarization.")
$tts.Dispose()
```

macOS / Linux

```bash
# System dep
brew install ffmpeg            # macOS
# or: sudo apt-get update && sudo apt-get install -y ffmpeg

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

# Optional CPU flags
export CT2_FORCE_CPU=1
export WHISPER_MODEL=medium

# Optional diarization
# export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX

streamlit run app/streamlit_app.py
```


Environment Variables

| Variable                          | Purpose                         | Example               |
| --------------------------------- | ------------------------------- | --------------------- |
| `WHISPER_MODEL`                   | Whisper size                    | `medium` / `large-v3` |
| `WHISPER_DEVICE`                  | Force device                    | `cpu` / `cuda`        |
| `CT2_FORCE_CPU`                   | Force CTranslate2 CPU           | `1`                   |
| `CUDA_VISIBLE_DEVICES`            | Hide GPUs                       | `""`                  |
| `HF_TOKEN`                        | Enable diarization (pyannote)   | `hf_…`                |
| `CHROMA_TELEMETRY`                | Silence Chroma telemetry        | `0`                   |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | Silence Windows symlink warning | `1`                   |

> You can also put `HF_TOKEN` in `.streamlit/secrets.toml`.

Using the App

1. Open **[http://localhost:8501](http://localhost:8501)**.
2. In the **sidebar**, upload `.mp3/.wav/.m4a` and give each a title.
3. Click **Process & Index** (resample → transcribe → optional diarize → chunk → embed → index).
4. Search in the main panel; results show episode, `MM:SS` range, speaker hints, and snippet.

Try: *“What is transfer learning?”*, *“Which vector database is used?”*, *“What does diarization mean?”*.

---

## Deployment

### Hugging Face Spaces

1. New **Space** → **Streamlit**
2. Push this repo
3. **Settings → Variables & secrets**: add `HF_TOKEN` (optional)
4. Run (switch Hardware to a small GPU if needed)

### Docker

```bash
docker build -t podcast-rag .
docker run --rm -it -p 7860:7860 -e HF_TOKEN=$HF_TOKEN podcast-rag
# open http://localhost:7860
```

Troubleshooting

* **ffmpeg not found** → project calls a bundled `imageio-ffmpeg` binary; ensure `requirements.txt` installed.
* **`pyarrow.PyExtensionType` error** → pin `pyarrow==14.0.2`.
* **`pypika.dialects` missing** → use `chromadb==0.5.5`, `pypika==0.48.9`.
* **CUDA/cuDNN DLL warnings (Windows)** → set `CT2_FORCE_CPU=1` and `CUDA_VISIBLE_DEVICES=""`.
* **No HF token** → diarization gracefully falls back to single speaker.

---

Security & Privacy

* Audio stays local (or within your Space).
* Vector DB persists under `storage/chroma/` (gitignored).
* Don’t commit raw audio or vector stores unless intended.
* Keep tokens in env vars or `.streamlit/secrets.toml`.

---

License & Credits

* **License:** MIT
* **Credits:** OpenAI Whisper (via faster-whisper / CTranslate2), pyannote.audio, Sentence-Transformers, ChromaDB, Streamlit
* **Author:** Raj Shekhar

---
