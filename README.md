🎙️ Audio-to-Text RAG for Podcast Search

Upload podcast episodes → transcribe (word timestamps) → (optional) diarize speakers → time-aware chunking → vector search across episodes → query with timestamped results in a clean Streamlit UI.

Built for local dev (Chroma, CPU-friendly), easy GPU upgrade, and one-click deploy to Hugging Face Spaces.

<img width="1918" height="965" alt="rag1" src="https://github.com/user-attachments/assets/483b8170-b224-4279-a52b-019fbbbe4dc4" />
<img width="1317" height="841" alt="rag2" src="https://github.com/user-attachments/assets/bb3fec99-d860-4022-8005-893341d865fd" />
<img width="417" height="850" alt="rag3" src="https://github.com/user-attachments/assets/631c6323-689e-4798-8273-37af88f50d5a" />



Table of Contents

Key Features

System Architecture

Repository Structure

Requirements

Quickstart

Windows (PowerShell, Python 3.11, CPU)

macOS / Linux

Environment Variables

Using the App

Pipeline Details

Transcription

Diarization

Alignment & Chunking

Embeddings & Index

Retrieval & (Optional) Reranking

Evaluation (RAGAS)

Deployment

Hugging Face Spaces

Docker

Customization

Troubleshooting

Security & Privacy

License

Acknowledgments

Key Features

Whisper (faster-whisper) for accurate transcription with word-level timestamps

Optional pyannote.audio diarization (“who spoke when”)

Time-aware chunking with overlap to preserve context

Vector search with Chroma (local, persistent on disk)

(Optional) CrossEncoder reranking for precision on tricky queries

Streamlit UI with episode titles, MM:SS ranges, and speaker hints

“Just works” on CPU with easy environment flags; GPU support is plug-and-play later

One-click deploy to Hugging Face Spaces (Streamlit)

System Architecture
Audio (.mp3/.wav/.m4a)
      │
      ▼
FFmpeg resample → 16k mono WAV
      │
      ▼
Whisper (faster-whisper) → words + timestamps
      │
      ├─► (optional) pyannote.audio diarization → speaker turns
      │
      ▼
Align words ↔ speakers → sentences → time-aware windows
      │
      ▼
Embeddings (sentence-transformers) → Chroma (persistent)
      │
      ▼
Query → kNN retrieve (± CrossEncoder rerank) → timestamped results in Streamlit

Repository Structure
podcast-rag/
├─ app/
│  ├─ streamlit_app.py         # UI + search
│  ├─ components.py            # tiny UI helpers
│  └─ prompts.py               # (for LLM answer templating, if used)
├─ pipeline/
│  ├─ ingest.py                # ffmpeg resample → whisper → (optional) diarize → save JSON
│  ├─ align.py                 # assign speakers to words + sentence building
│  ├─ chunk.py                 # time-aware windowing with overlap
│  ├─ embed_index.py           # embeddings + Chroma upsert
│  └─ retrieve.py              # retrieval + optional CrossEncoder rerank
├─ storage/
│  ├─ chroma/                  # vector DB (persisted; gitignored)
│  └─ data/                    # uploaded audio + episode JSON
├─ eval/
│  └─ dataset.jsonl            # (optional) RAG evaluation sets
├─ requirements.txt
├─ Dockerfile
└─ README.md

Requirements

Python 3.11 (recommended for Windows; stable wheels for ML/audio stacks)

FFmpeg (CLI) – or imageio-ffmpeg is used automatically via code (no system install needed)

CPU is fine; GPU later requires a matching CUDA/cuDNN stack

If you’re on Windows, the project includes fixes to avoid common pitfalls (pyarrow version pin, optional diarization, ffmpeg path resolution, etc).

Quickstart
Windows (PowerShell, Python 3.11, CPU)
# 0) Go to your project dir
cd D:\podcast-rag

# 1) Create venv (Python 3.11)
py -3.11 -m venv .venv

# If activation policy blocks scripts (one-time per terminal):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate (optional; you can also always use .\.venv\Scripts\python)
.\.venv\Scripts\Activate

# 2) Install deps
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

# 3) (Optional) install FFmpeg system-wide OR skip; code uses imageio-ffmpeg internally
# winget install Gyan.FFmpeg
# ffmpeg -version

# 4) CPU-friendly env flags (recommended)
$Env:CT2_FORCE_CPU = "1"         # force faster-whisper to CPU
$Env:CUDA_VISIBLE_DEVICES = ""   # hide GPUs to avoid cuDNN errors
$Env:WHISPER_MODEL = "medium"    # faster CPU testing; use "large-v3" for accuracy

# 5) (Optional) diarization token – enables pyannote
# $Env:HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"

# 6) Run the app
python -m streamlit run app\streamlit_app.py


PowerShell TTS (make a quick test WAV):

Add-Type -AssemblyName System.Speech
$tts = New-Object System.Speech.Synthesis.SpeechSynthesizer
$tts.SetOutputToWaveFile("episode1.wav")
$tts.Speak("This is a quick test of our podcast RAG system. We mention transfer learning, Chroma, and diarization.")
$tts.Dispose()

macOS / Linux
# 0) System dep
brew install ffmpeg         # macOS (or) sudo apt-get install -y ffmpeg  # Ubuntu

# 1) Create venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

# 3) CPU mode (optional)
export CT2_FORCE_CPU=1
export WHISPER_MODEL=medium

# 4) (Optional) diarization token
# export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX

# 5) Run
streamlit run app/streamlit_app.py

Environment Variables
Variable	Purpose	Example
WHISPER_MODEL	Whisper size (medium, large-v3, etc.)	medium
WHISPER_DEVICE	Optional override (cpu/cuda)	cpu
CT2_FORCE_CPU	Force CTranslate2 to use CPU	1
CUDA_VISIBLE_DEVICES	Hide GPUs for CPU-only runs	""
HF_TOKEN	Hugging Face token for pyannote diarization	hf_XXXXXXXXXXXXXXXXXXXXXXXX
CHROMA_TELEMETRY	Silence Chroma telemetry warnings	0
HF_HUB_DISABLE_SYMLINKS_WARNING	Silence Windows symlink cache warnings	1

You can also set HF_TOKEN in .streamlit/secrets.toml and the app will pick it up.

Using the App

Open the Streamlit UI (http://localhost:8501).

In the sidebar, upload one or more .mp3/.wav/.m4a files and give each a title.

Click Process & Index:

Convert audio → 16 kHz mono WAV (via imageio-ffmpeg)

Transcribe with Whisper (word timestamps)

(Optional) Diarize with pyannote (if HF_TOKEN present)

Build time-aware windows (overlap for context)

Embed and upsert into Chroma

Use the search box (main panel). Results show:

Episode title

MM:SS start–end range

Speakers (if diarization enabled)

Snippet text

Suggested queries:

“What is transfer learning?”

“Which vector database is used?”

“How big is each chunk?”

“What does diarization mean?”

Pipeline Details
Transcription

Library: faster-whisper

Default model: medium (CPU-friendly); change with WHISPER_MODEL

Output: word list with text, start, end seconds

Diarization

Library: pyannote.audio (optional)

Requires HF_TOKEN (and agreement on model page)

When unavailable, the pipeline gracefully falls back to a single speaker.

Alignment & Chunking

Words are assigned to speakers based on time overlap.

Sentences are constructed with punctuation and max char length caps.

Windows target ~400 tokens with ~20% overlap to preserve context across boundaries.

Embeddings & Index

Embeddings: sentence-transformers/all-MiniLM-L6-v2

DB: Chroma (persistent at storage/chroma/)

Metadata: episode id/title, start_time, end_time, tokens, n_speakers, top_speaker, and speakers_json (stringified)

Retrieval & (Optional) Reranking

kNN from Chroma → top-k candidate chunks

CrossEncoder (ms-marco-MiniLM-L-6-v2) rerank toggle in UI for improved precision

Evaluation (RAGAS)

You can evaluate with RAGAS using a small dataset of (question, answer, context) in eval/dataset.jsonl. Example:

{"question": "What is transfer learning?", "answer": "…", "contexts": ["…", "…"]}


Basic loop (pseudo):

from ragas import evaluate
# build dataset dict with "question", "answer", "contexts"
score = evaluate(dataset)
print(score)

Deployment
Hugging Face Spaces

Create a New Space → Streamlit.

Push the repo files.

In Settings → Variables and secrets, add:

HF_TOKEN (for diarization; optional)

Spaces will install from requirements.txt and auto-launch.

If latency is high, switch Hardware to a small GPU.

Notes for Spaces:

CPU works; use WHISPER_MODEL=medium for speed.

Very large uploads can hit free-tier timeouts.

Docker
docker build -t podcast-rag .
docker run --rm -it -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  podcast-rag
# Open http://localhost:7860

Customization

Faster Whisper model: set WHISPER_MODEL to large-v3 for max accuracy (slower on CPU).

Device: set WHISPER_DEVICE=cpu|cuda; default is CPU in this template.

Vector DB: swap Chroma for Pinecone/Weaviate by replacing:

pipeline/embed_index.py (client + upserts)

pipeline/retrieve.py (queries/filters)

Reranking: default off in UI; enable per search if you prefer precision.

Chunking: tune target tokens & overlap in pipeline/chunk.py.

Troubleshooting

“FileNotFoundError in ffmpeg”

The project calls ffmpeg by absolute path via imageio-ffmpeg — no system install needed.

If you edited code and reintroduced ffmpeg-python, either install FFmpeg (winget install Gyan.FFmpeg) or restore the bundled call in ingest.py.

Windows: ModuleNotFoundError: pypika.dialects

Use pinned versions: chromadb==0.5.5, pypika==0.48.9.

Windows: AttributeError: pyarrow.PyExtensionType

Pin pyarrow==14.0.2.

Diarization assertion / HF_TOKEN

Either set a token (HF_TOKEN=hf_…) or rely on the single-speaker fallback (default).

CUDA/cuDNN DLL errors (Windows)

Force CPU: set CT2_FORCE_CPU=1 and CUDA_VISIBLE_DEVICES="".

PowerShell script activation blocked

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Slow on CPU

Use WHISPER_MODEL=medium, keep reranking off, or try GPU hardware (Spaces).

Security & Privacy

Audio is processed locally in your environment (or your Space).

Vector DB lives in storage/chroma/ (gitignored).

Do not commit raw audio or vector stores unless you intend to share them.

Keep tokens (e.g., HF_TOKEN) in environment variables or .streamlit/secrets.toml:

HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"

License

MIT

Acknowledgments

OpenAI Whisper (via faster-whisper / CTranslate2)

pyannote.audio for robust diarization

Sentence Transformers for embeddings

ChromaDB for easy local vector search

Streamlit for the UI

Made by Raj Shekhar

