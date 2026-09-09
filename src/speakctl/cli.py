"""`speak` — turn one or more snippets of text into audio files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import DEFAULT_MODEL, DEFAULT_VOICE, Job, TTSEngine


def _parse_job(spec: str, default_voice: str) -> Job:
    """Parse a `voice:text` pair; a bare snippet falls back to the default voice."""
    voice, sep, text = spec.partition(":")
    if not sep or not voice.strip():
        return Job(text=spec, voice=default_voice)
    return Job(text=text, voice=voice.strip())


def _collect_jobs(args: argparse.Namespace) -> list[Job]:
    jobs = [_parse_job(spec, args.voice) for spec in args.job]
    jobs += [Job(text=text, voice=args.voice) for text in args.text]
    if not jobs and not sys.stdin.isatty():
        jobs = [_parse_job(line, args.voice) for line in sys.stdin.read().splitlines() if line.strip()]
    return [job for job in jobs if job.text.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="speak",
        description="Speak text locally with MLX-Audio. Pair snippets with voices "
        "using -j VOICE:TEXT; every snippet is rendered in one process, "
        "so the model loads only once.",
    )
    parser.add_argument("text", nargs="*", help="text to speak, in the default voice")
    parser.add_argument(
        "-j",
        "--job",
        action="append",
        default=[],
        metavar="VOICE:TEXT",
        help="a snippet and the voice to speak it in (repeatable)",
    )
    parser.add_argument("-o", "--output", help="output file (single snippet only)")
    parser.add_argument("-d", "--outdir", default=".", help="directory for generated files (default: .)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"default voice (default: {DEFAULT_VOICE})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MLX-Audio model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--speed", type=float, default=1.0, help="speech rate multiplier (default: 1.0)")
    args = parser.parse_args()

    jobs = _collect_jobs(args)
    if not jobs:
        parser.error("no text given: pass text, -j VOICE:TEXT, or pipe it on stdin")
    if args.output:
        if len(jobs) > 1:
            parser.error("-o works with a single snippet; use --outdir for several")
        jobs[0].output_path = Path(args.output)

    engine = TTSEngine(model_id=args.model, speed=args.speed)
    for path in engine.synthesize_many(jobs, outdir=args.outdir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
