"""Text-to-speech engine: turns text into 16-bit PCM using a pluggable backend.

This module imports only the stdlib and numpy. The MLX model lives in
`mlx_backend.py` and is imported lazily, so tests and clients can use the
engine API without MLX installed.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Protocol

import numpy as np

from .protocol import DEFAULT_VOICE

DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_SAMPLE_RATE = 24000
# Sentence-sized chunks keep the time to first audio low when streaming.
DEFAULT_SPLIT_PATTERN = r"(?<=[.!?…])\s+|\n+"
WARMUP_TEXT = "Warming up."


class Backend(Protocol):
    """Something that renders text to float32 mono audio, one chunk at a time."""

    sample_rate: int

    def render(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        """Yield 1-D float32 arrays with samples in [-1, 1]."""
        ...


@dataclass
class Job:
    """One snippet of text with the voice it should be spoken in."""

    text: str
    voice: str = DEFAULT_VOICE
    output_path: Path | None = None


def float_to_s16le(samples: np.ndarray) -> bytes:
    """Convert float audio in [-1, 1] to little-endian signed 16-bit PCM bytes."""
    audio = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (audio * 32767).astype("<i2").tobytes()


class TTSEngine:
    """Loads a TTS backend once and synthesizes any number of snippets with it.

    `split_pattern` is the regex the MLX model splits text on before rendering;
    None keeps the model's own default. It only applies when the engine loads
    the model itself, and is ignored when an explicit `backend` is passed.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        speed: float = 1.0,
        backend: Backend | None = None,
        split_pattern: str | None = DEFAULT_SPLIT_PATTERN,
    ):
        self.model_id = model_id
        self.speed = speed
        if backend is None:
            from .mlx_backend import MlxBackend

            backend = MlxBackend.load(model_id, split_pattern)
        self.backend = backend

    @property
    def sample_rate(self) -> int:
        return int(self.backend.sample_rate)

    def stream_pcm(self, text: str, voice: str = DEFAULT_VOICE, speed: float | None = None) -> Iterator[bytes]:
        """Yield s16le mono PCM, one piece per chunk the backend produces."""
        text = text.strip()
        if not text:
            raise ValueError("no text to synthesize")
        speed = self.speed if speed is None else speed
        produced = False
        for chunk in self.backend.render(text, voice, speed):
            pcm = float_to_s16le(chunk)
            if not pcm:
                continue
            produced = True
            yield pcm
        if not produced:
            raise RuntimeError(f"model produced no audio for: {text[:60]!r}")

    def synthesize_pcm(self, text: str, voice: str = DEFAULT_VOICE, speed: float | None = None) -> bytes:
        """Render `text` to one block of s16le mono PCM."""
        return b"".join(self.stream_pcm(text, voice=voice, speed=speed))

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice: str = DEFAULT_VOICE,
        speed: float | None = None,
    ) -> Path:
        """Render `text` in `voice` to a WAV file at `output_path` and return the path."""
        chunks = self.stream_pcm(text, voice=voice, speed=speed)
        first = next(chunks)  # validates text and surfaces errors before touching disk
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(first)
            for pcm in chunks:
                wav.writeframes(pcm)
        return path

    def synthesize_many(self, jobs: List[Job], outdir: str | Path = ".") -> List[Path]:
        """Render every job, reusing the one loaded model. Returns paths in order."""
        outdir = Path(outdir)
        paths = []
        for index, job in enumerate(jobs, start=1):
            path = job.output_path or outdir / f"{index:03d}_{job.voice}.wav"
            paths.append(self.synthesize(job.text, path, voice=job.voice))
        return paths

    def warmup(self) -> None:
        """Run one short synthesis so the first real request doesn't pay compile costs."""
        self.synthesize_pcm(WARMUP_TEXT)
