import os
import imageio_ffmpeg as iio_ffm
os.environ.setdefault("FFMPEG_BINARY", iio_ffm.get_ffmpeg_exe())

import json
import uuid
import tempfile
from pathlib import Path
import subprocess

from faster_whisper import WhisperModel


AUDIO_DIR = Path("storage/data")
JSON_DIR = Path("storage/data")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(input_path: Path) -> str:
    """Convert any input to 16 kHz mono WAV and return the temp WAV path."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg_exe = iio_ffm.get_ffmpeg_exe()  # absolute path to ffmpeg.exe

    # Call ffmpeg directly by absolute path (no PATH or env var required)
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        tmp_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg not found at: {ffmpeg_exe}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed converting {input_path} -> {tmp_wav}") from e

    return tmp_wav


def transcribe_with_whisper(wav_path: str, device: str | None = None, model_size: str | None = None):
    import os
    device = (device or os.getenv("WHISPER_DEVICE") or "cpu").lower()
    model_size = model_size or os.getenv("WHISPER_MODEL", "medium")

    # If CPU, tell CTranslate2 to not even try CUDA
    if device == "cpu":
        os.environ["CT2_FORCE_CPU"] = "1"

    model = WhisperModel(model_size, device=device, compute_type="auto")
    segments, info = model.transcribe(
        wav_path, beam_size=5, word_timestamps=True, vad_filter=True
    )
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"text": w.word, "start": float(w.start), "end": float(w.end)})
    return words, getattr(info, "language", "en")


def diarize_with_pyannote(wav_path: str, hf_token_env: str = "HF_TOKEN"):
    """
    Optional diarization.
    Returns a list of turns: [{"speaker": str, "start": float, "end": float}, ...]
    If token missing or any error occurs, returns [] (single-speaker fallback will be used).
    """
    token = os.getenv(hf_token_env)
    if not token:
        return []
    try:
        # Import inside function so the module isn’t required if diarization is off
        from pyannote.audio import Pipeline as DiarizationPipeline
        pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        diarization = pipeline({"audio": wav_path})
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({"speaker": speaker, "start": float(turn.start), "end": float(turn.end)})
        return sorted(turns, key=lambda x: x["start"])
    except Exception as e:
        print(f"[diarization disabled] {e}")
        return []


def save_episode_json(episode_title: str, wav_path: str, words: list, turns: list, language: str):
    """Persist episode JSON and return (json_path, episode_id)."""
    episode_id = str(uuid.uuid4())[:8]
    out = {
        "episode_id": episode_id,
        "episode_title": episode_title,
        "audio_path": wav_path,
        "language": language,
        "words": words,
        "turns": turns,
    }
    out_path = JSON_DIR / f"{episode_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out_path, episode_id


def process_episode(file_path: Path, title: str):
    """
    Full pipeline for one file: convert -> transcribe -> (optional) diarize -> save JSON.
    Falls back to a single speaker if diarization is unavailable.
    Returns (episode_json_path, episode_id).
    """
    wav = load_audio(file_path)
    words, lang = transcribe_with_whisper(wav)
    turns = diarize_with_pyannote(wav)

    if not turns:
        # single-speaker fallback covering the clip
        if words:
            turns = [{"speaker": "SPK0", "start": words[0]["start"], "end": words[-1]["end"]}]
        else:
            turns = [{"speaker": "SPK0", "start": 0.0, "end": 0.0}]

    return save_episode_json(title, wav, words, turns, lang)
