"""speakctl: local text-to-speech.

`SpeakClient` and the protocol constants import with the stdlib alone, so a
client app needs neither numpy nor MLX. Engine names load on first use.
"""

from .client import SpeakClient, SpeakError
from .protocol import DEFAULT_VOICE

__all__ = [
    "TTSEngine",
    "Job",
    "Backend",
    "SpeakClient",
    "SpeakError",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "DEFAULT_SAMPLE_RATE",
]

_ENGINE_NAMES = {"TTSEngine", "Job", "Backend", "DEFAULT_MODEL", "DEFAULT_SAMPLE_RATE"}


def __getattr__(name):
    if name in _ENGINE_NAMES:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
