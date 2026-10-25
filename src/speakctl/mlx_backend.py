"""MLX-Audio backend. The only module that imports mlx or huggingface_hub."""

from __future__ import annotations

import inspect
import os
from typing import Iterator

import numpy as np

from .engine import DEFAULT_SAMPLE_RATE


def _load_model_cache_first(model_id: str):
    """Try the local cache with no network round-trip; fetch only if something's missing."""
    from mlx_audio.tts.utils import load_model
    from huggingface_hub.errors import LocalEntryNotFoundError
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    prior = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return load_model(model_id)
    except (LocalEntryNotFoundError, FileNotFoundError):
        pass
    finally:
        if prior is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior
    return load_model(model_id)


class MlxBackend:
    """Adapts an MLX-Audio TTS model to the engine's `Backend` protocol.

    MLX default streams are thread-local: load and render on the same thread.
    """

    def __init__(self, model, split_pattern: str | None = None):
        self.model = model
        self.split_pattern = split_pattern
        self.sample_rate = int(getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE))
        self._supported = set(inspect.signature(model.generate).parameters)

    @classmethod
    def load(cls, model_id: str, split_pattern: str | None = None) -> "MlxBackend":
        return cls(_load_model_cache_first(model_id), split_pattern)

    def render(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        import mlx.core as mx

        kwargs = {"text": text, "voice": voice}
        optional = {"speed": speed, "lang_code": voice[0], "split_pattern": self.split_pattern}
        for name, value in optional.items():
            if name in self._supported and value is not None:
                kwargs[name] = value

        for result in self.model.generate(**kwargs):
            # numpy has no bfloat16, so cast inside MLX before converting.
            audio = result.audio.astype(mx.float32)
            rate = getattr(result, "sample_rate", None)
            if rate is not None and int(rate) != self.sample_rate:
                raise RuntimeError(f"model returned {rate} Hz audio, expected {self.sample_rate} Hz")
            yield np.array(audio).reshape(-1)
