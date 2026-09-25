"""`speak` turns snippets of text into audio files; `speak-serve` streams audio over a socket."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .engine import DEFAULT_MODEL, DEFAULT_SPLIT_PATTERN, DEFAULT_VOICE, Job, TTSEngine


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
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep the model loaded and synthesize one voice:text line per stdin "
        "line as it arrives, instead of exiting after one batch",
    )
    args = parser.parse_args()

    if args.watch:
        engine = TTSEngine(model_id=args.model, speed=args.speed, split_pattern=None)
        outdir = Path(args.outdir)
        for index, line in enumerate(sys.stdin, start=1):
            job = _parse_job(line.strip(), args.voice)
            if not job.text.strip():
                continue
            path = outdir / f"{index:03d}_{job.voice}.wav"
            engine.synthesize(job.text, path, voice=job.voice)
            print(path, flush=True)
        return 0

    jobs = _collect_jobs(args)
    if not jobs:
        parser.error("no text given: pass text, -j VOICE:TEXT, or pipe it on stdin")
    if args.output:
        if len(jobs) > 1:
            parser.error("-o works with a single snippet; use --outdir for several")
        jobs[0].output_path = Path(args.output)

    engine = TTSEngine(model_id=args.model, speed=args.speed, split_pattern=None)
    for path in engine.synthesize_many(jobs, outdir=args.outdir):
        print(path)
    return 0


def serve_main(argv: list[str] | None = None) -> int:
    from .server import DEFAULT_HOST, DEFAULT_PORT, Limits, SpeakServer

    parser = argparse.ArgumentParser(
        prog="speak-serve",
        description="Keep the model loaded and stream speech to local clients over "
        "TCP or a Unix socket. See the README for the wire protocol.",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--host", help=f"TCP host to bind (default: {DEFAULT_HOST})")
    where.add_argument("--socket", metavar="PATH", help="listen on a Unix socket instead of TCP")
    parser.add_argument("--port", type=int, help=f"TCP port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MLX-Audio model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--speed", type=float, default=1.0, help="default speech rate multiplier (default: 1.0)")
    parser.add_argument(
        "--split-pattern",
        default=DEFAULT_SPLIT_PATTERN,
        help="regex the model splits text on; each piece is streamed as soon as it's ready "
        "(default: sentence boundaries)",
    )
    parser.add_argument(
        "--max-text-chars", type=int, default=Limits.max_text_chars, help="longest accepted request text"
    )
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up synthesis at startup")
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    if args.socket and args.port is not None:
        parser.error("argument --port: not allowed with argument --socket")

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("speakctl.serve")

    if args.socket:
        address = args.socket
    else:
        address = (args.host or DEFAULT_HOST, DEFAULT_PORT if args.port is None else args.port)

    server = SpeakServer(
        address,
        engine_factory=lambda: TTSEngine(model_id=args.model, speed=args.speed, split_pattern=args.split_pattern),
        limits=Limits(max_text_chars=args.max_text_chars),
        warmup=not args.no_warmup,
    )
    try:
        log.info("loading %s", args.model)
        server.start()
        where_desc = server.server_address if args.socket else "%s:%d" % server.server_address[:2]
        log.info("ready, listening on %s", where_desc)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
