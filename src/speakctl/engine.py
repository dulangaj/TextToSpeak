"""Text-to-speech engine wrapping MLX-Audio. The only place model logic lives."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "af_heart"
DEFAULT_SAMPLE_RATE = 24000


@dataclass
class Job:
    """One snippet of text with the voice it should be spoken in."""

    text: str
    voice: str = DEFAULT_VOICE
    output_path: Path | None = None


class TTSEngine:
    """Loads a TTS model once and synthesizes any number of snippets with it."""

    def __init__(self, model_id: str = DEFAULT_MODEL, speed: float = 1.0):
        self.model_id = model_id
        self.speed = speed
        self.model = load_model(model_id)
        self._accepts_lang_code = "lang_code" in inspect.signature(self.model.generate).parameters

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice: str = DEFAULT_VOICE,
        speed: float | None = None,
    ) -> Path:
        """Render `text` in `voice` to `output_path` and return the path written."""
        text = text.strip()
        if not text:
            raise ValueError("no text to synthesize")

        kwargs = {"lang_code": voice[0]} if self._accepts_lang_code else {}
        results = list(
            self.model.generate(
                text=text,
                voice=voice,
                speed=self.speed if speed is None else speed,
                **kwargs,
            )
        )
        if not results:
            raise RuntimeError(f"model produced no audio for: {text[:60]!r}")

        audio = mx.concatenate([r.audio for r in results])
        sample_rate = getattr(results[0], "sample_rate", DEFAULT_SAMPLE_RATE)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio_write(path, audio, sample_rate)
        return path

    def synthesize_many(self, jobs: list[Job], outdir: str | Path = ".") -> list[Path]:
        """Render every job, reusing the one loaded model. Returns paths in order."""
        outdir = Path(outdir)
        paths = []
        for index, job in enumerate(jobs, start=1):
            path = job.output_path or outdir / f"{index:03d}_{job.voice}.wav"
            paths.append(self.synthesize(job.text, path, voice=job.voice))
        return paths
