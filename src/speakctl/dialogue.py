"""`texttospeak` renders a multi-voice conversation to one WAV file.

Callers write lines as `SPEAKER: text`, where SPEAKER is an alias (voice1 to
voice4) or any Kokoro voice name, and get back a single file with every line
in order and a short pause between them.
"""

from __future__ import annotations

import argparse
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .engine import DEFAULT_MODEL, TTSEngine
from .voices import KOKORO_VOICES

DEFAULT_CAST = {
    "voice1": "af_heart",
    "voice2": "am_michael",
    "voice3": "bf_emma",
    "voice4": "bm_george",
}
DEFAULT_SPEAKER = "voice1"
DEFAULT_OUTPUT = "conversation.wav"
DEFAULT_PAUSE = 0.35

VOICE_NAME = re.compile(r"^[a-z]{2}_[a-z0-9]+$")
# Only a single word before the first colon counts as a speaker, so text like
# "Meet at 10:30" stays bare text.
SPEAKER_PREFIX = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:(.*)$", re.DOTALL)


@dataclass(frozen=True)
class Turn:
    """One line of the conversation: who says it (as written), the voice, and the text."""

    speaker: str
    voice: str
    text: str


def resolve_cast(spec: str | None) -> Dict[str, str]:
    """Merge `voice1=af_heart,voice2=...` overrides onto the default cast."""
    cast = dict(DEFAULT_CAST)
    if not spec:
        return cast
    for item in spec.split(","):
        if not item.strip():
            continue
        alias, sep, voice = item.partition("=")
        alias, voice = alias.strip().lower(), voice.strip().lower()
        if not sep or alias not in DEFAULT_CAST:
            aliases = ", ".join(DEFAULT_CAST)
            raise ValueError(f"bad --cast entry {item.strip()!r}: expected ALIAS=VOICE with ALIAS one of {aliases}")
        if not VOICE_NAME.match(voice):
            raise ValueError(f"bad --cast entry {item.strip()!r}: {voice!r} is not a voice name")
        cast[alias] = voice
    return cast


def parse_line(line: str, cast: Dict[str, str]) -> Turn:
    """Parse `SPEAKER: text` or bare text (spoken by voice1). Raises ValueError on an unknown speaker."""
    match = SPEAKER_PREFIX.match(line)
    if not match:
        return Turn(DEFAULT_SPEAKER, cast[DEFAULT_SPEAKER], line.strip())
    speaker, text = match.group(1), match.group(2).strip()
    key = speaker.lower()
    if key in cast:
        return Turn(speaker, cast[key], text)
    if VOICE_NAME.match(key):
        return Turn(speaker, key, text)
    raise ValueError(f"unknown speaker {speaker!r}: use one of {', '.join(cast)} or a voice name like am_michael")


def build_jobs(lines: Iterable[str], cast: Dict[str, str]) -> List[Turn]:
    """Parse every line in order, dropping blank lines and lines with no text."""
    turns = []
    for line in lines:
        if not line.strip():
            continue
        turn = parse_line(line, cast)
        if turn.text:
            turns.append(turn)
    return turns


def _write_wav(path: Path, pcm_pieces: Iterable[bytes], sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for pcm in pcm_pieces:
            wav.writeframes(pcm)
    return path


def _list_voices(cast: Dict[str, str]) -> None:
    for alias, voice in cast.items():
        print(f"{alias}  {voice}")
    print()
    for voice in KOKORO_VOICES:
        print(voice)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="texttospeak",
        description="Render a conversation to one WAV file. Each LINE is 'SPEAKER: text', where SPEAKER "
        "is voice1 to voice4 or a Kokoro voice name; bare text is spoken by voice1. With no LINE "
        "arguments, lines are read from stdin.",
    )
    parser.add_argument("lines", nargs="*", metavar="LINE", help="a line of the conversation")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help=f"output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--split", action="store_true", help="also write one file per line into --outdir")
    parser.add_argument("--outdir", help="directory for --split files (default: next to the output)")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE,
                        help=f"seconds of silence between lines (default: {DEFAULT_PAUSE})")
    parser.add_argument("--speed", type=float, default=1.0, help="speech rate multiplier (default: 1.0)")
    parser.add_argument("--cast", metavar="ALIAS=VOICE,...",
                        help="remap aliases, e.g. voice1=bf_alice,voice2=am_adam")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MLX-Audio model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--list-voices", action="store_true", help="print the aliases and every voice name, then exit")
    args = parser.parse_args(argv)

    try:
        cast = resolve_cast(args.cast)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list_voices:
        _list_voices(cast)
        return 0
    if args.pause < 0:
        parser.error("--pause must be zero or more")
    if args.speed <= 0:
        parser.error("--speed must be more than zero")

    lines = args.lines
    if not lines and not sys.stdin.isatty():
        lines = sys.stdin.read().splitlines()
    try:
        turns = build_jobs(lines, cast)
    except ValueError as exc:
        parser.error(str(exc))
    if not turns:
        parser.error("no lines given: pass 'SPEAKER: text' arguments or pipe them on stdin")

    output = Path(args.output)
    outdir = Path(args.outdir) if args.outdir else output.parent
    try:
        engine = TTSEngine(model_id=args.model, speed=args.speed, split_pattern=None)
        rendered = [b"".join(engine.stream_pcm(turn.text, voice=turn.voice)) for turn in turns]
        rate = engine.sample_rate
        silence = b"\x00\x00" * round(args.pause * rate)
        pieces = []
        for index, pcm in enumerate(rendered):
            if index and silence:
                pieces.append(silence)
            pieces.append(pcm)
        paths = [_write_wav(output, pieces, rate)]
        if args.split:
            for index, (turn, pcm) in enumerate(zip(turns, rendered), start=1):
                paths.append(_write_wav(outdir / f"{index:03d}_{turn.speaker}.wav", [pcm], rate))
    except Exception as exc:  # one-line message for callers, not a traceback
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"texttospeak: error: {message}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
