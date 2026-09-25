"""Long-lived TTS server: load the model once, stream PCM to many clients.

One inference worker thread owns the engine and renders requests in arrival
order. Each connection has a reader (the socketserver handler thread) and a
writer thread that drains a bounded outbox, so a slow client never blocks
another client's reads. See protocol.py for the wire format.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import socket
import socketserver
import stat
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple, Union

from . import protocol as p
from .engine import TTSEngine

log = logging.getLogger(__name__)

VOICE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SPEED_MIN, SPEED_MAX = 0.5, 2.0
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7333

_STOP = object()


@dataclass
class Limits:
    max_text_chars: int = 10_000
    max_inflight: int = 32  # per connection: queued + rendering
    outbox_frames: int = 256  # per connection, before send() starts blocking
    send_timeout: float = 30.0  # seconds a stuck client may block the worker


@dataclass
class Request:
    conn: "Connection"
    request_id: int
    spec: p.RequestSpec
    cancelled: threading.Event = field(default_factory=threading.Event)


class Connection:
    """One client socket: a bounded outbox drained by a writer thread, plus in-flight requests."""

    def __init__(self, sock: socket.socket, limits: Limits):
        self.sock = sock
        self.limits = limits
        self.inflight: Dict[int, Request] = {}
        self.lock = threading.Lock()
        self.closed = False
        self._outbox: "queue.Queue[object]" = queue.Queue(maxsize=limits.outbox_frames)
        self._writer = threading.Thread(target=self._write_loop, name="speak-writer", daemon=True)
        self._writer.start()

    def send(self, frame: p.Frame) -> None:
        """Queue a frame for the client. A no-op once the connection is closed."""
        deadline = time.monotonic() + self.limits.send_timeout
        while not self.closed:
            try:
                # Short waits so a close() from another thread releases us promptly.
                self._outbox.put(frame, timeout=0.1)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    log.warning("client not reading for %gs; dropping connection", self.limits.send_timeout)
                    self.close(abort=True)

    def close(self, abort: bool = False) -> None:
        """Stop accepting frames and cancel in-flight work. Idempotent.

        Normally frames already queued are flushed before the socket is shut
        down; `abort=True` shuts it down immediately.
        """
        with self.lock:
            first = not self.closed
            self.closed = True
            for req in self.inflight.values():
                req.cancelled.set()
        if first:
            try:
                self._outbox.put_nowait(_STOP)
            except queue.Full:
                abort = True
        if abort:
            self._shutdown()

    def join(self) -> None:
        """Wait for queued frames to flush; force the socket down if the client won't read."""
        self._writer.join(self.limits.send_timeout)
        if self._writer.is_alive():
            self._shutdown()
            self._writer.join()

    def finish(self, req: Request, frame: p.Frame) -> None:
        """Retire a request, then send its terminal frame.

        Popping first means the client may reuse the id as soon as it sees END/ERROR.
        """
        with self.lock:
            self.inflight.pop(req.request_id, None)
        self.send(frame)

    def cancel(self, request_id: int) -> None:
        with self.lock:
            req = self.inflight.get(request_id)
        if req is not None:
            req.cancelled.set()

    def _shutdown(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _write_loop(self) -> None:
        while True:
            frame = self._outbox.get()
            if frame is _STOP:
                break
            try:
                self.sock.sendall(p.encode(frame))  # type: ignore[arg-type]
            except OSError:
                self.close(abort=True)
                return
        self._shutdown()


class Handler(socketserver.BaseRequestHandler):
    server: "SpeakServer"

    def handle(self) -> None:
        server = self.server
        server.wait_ready()
        conn = Connection(self.request, server.limits)
        server.track(conn, True)
        try:
            conn.send(p.hello_frame(server.hello_info()))
            self._serve(conn)
        finally:
            conn.close()
            conn.join()
            server.track(conn, False)

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining:
            try:
                data = self.request.recv(remaining)
            except OSError:
                data = b""
            if not data:
                break
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _serve(self, conn: Connection) -> None:
        request_id: Optional[int] = None
        try:
            while not conn.closed:
                frame = p.read_frame(self._read_exact)
                if frame is None:
                    return
                request_id = frame.request_id
                if frame.type == p.FrameType.REQUEST:
                    self._on_request(conn, frame)
                elif frame.type == p.FrameType.CANCEL:
                    conn.cancel(frame.request_id)
                else:
                    raise p.ProtocolError(f"{frame.type.name} frames are server-to-client only")
        except p.ProtocolError as exc:
            log.info("protocol error from client: %s", exc)
            if request_id is not None:
                conn.send(p.error_frame(request_id, p.ErrorCode.PROTOCOL_ERROR, str(exc)))

    def _on_request(self, conn: Connection, frame: p.Frame) -> None:
        rid = frame.request_id
        limits = self.server.limits

        def reject(code: str, message: str) -> None:
            conn.send(p.error_frame(rid, code, message))

        if rid == 0:
            return reject(p.ErrorCode.BAD_REQUEST, "request_id 0 is reserved")
        try:
            spec = p.parse_request(frame)
        except p.ProtocolError as exc:
            # The frame itself was well-formed, so the stream is still in sync.
            return reject(p.ErrorCode.BAD_REQUEST, str(exc))
        if not spec.text.strip():
            return reject(p.ErrorCode.BAD_REQUEST, "text is empty")
        if len(spec.text) > limits.max_text_chars:
            return reject(p.ErrorCode.BAD_REQUEST, f"text exceeds {limits.max_text_chars} characters")
        if not VOICE_RE.match(spec.voice):
            return reject(p.ErrorCode.BAD_REQUEST, f"invalid voice name: {spec.voice!r}")
        if spec.speed is not None and not SPEED_MIN <= spec.speed <= SPEED_MAX:
            return reject(p.ErrorCode.BAD_REQUEST, f"speed must be between {SPEED_MIN} and {SPEED_MAX}")
        with conn.lock:
            if rid in conn.inflight:
                return reject(p.ErrorCode.BAD_REQUEST, f"request_id {rid} is already in flight")
            if len(conn.inflight) >= limits.max_inflight:
                return reject(p.ErrorCode.TOO_MANY_REQUESTS, f"at most {limits.max_inflight} requests in flight")
            req = Request(conn, rid, spec)
            conn.inflight[rid] = req
        self.server.submit(req)


Address = Union[Tuple[str, int], str]


class SpeakServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Binds on construction; call start() to load the model, then serve_forever()."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: Address,
        engine_factory: Callable[[], TTSEngine],
        limits: Limits | None = None,
        warmup: bool = True,
    ):
        self.limits = limits or Limits()
        self.engine_factory = engine_factory
        self.warmup = warmup
        self.engine: Optional[TTSEngine] = None
        self.unix_path: Optional[str] = None
        self._jobs: "queue.Queue[object]" = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._worker: Optional[threading.Thread] = None
        self._conns: Set[Connection] = set()
        self._conns_lock = threading.Lock()
        if isinstance(address, str):
            self.address_family = socket.AF_UNIX
            self.unix_path = address
            _remove_stale_socket(address)
        super().__init__(address, Handler)

    def server_bind(self) -> None:
        if self.unix_path is not None:
            self.socket.bind(self.unix_path)
            self.server_address = self.unix_path
        else:
            super().server_bind()

    # -- lifecycle -------------------------------------------------------

    def start(self, timeout: float | None = None) -> None:
        """Load the engine on the inference thread and block until it's ready."""
        if self._worker is None:
            self._worker = threading.Thread(target=self._run_worker, name="speak-inference", daemon=True)
            self._worker.start()
        self.wait_ready(timeout)

    def wait_ready(self, timeout: float | None = None) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError("engine did not become ready in time")
        if self._startup_error is not None:
            raise self._startup_error

    def server_close(self) -> None:
        self._jobs.put(_STOP)
        with self._conns_lock:
            conns = list(self._conns)
        for conn in conns:
            conn.close(abort=True)
        super().server_close()
        if self.unix_path is not None:
            try:
                os.unlink(self.unix_path)
            except FileNotFoundError:
                pass

    # -- used by handlers ------------------------------------------------

    def hello_info(self) -> dict:
        assert self.engine is not None
        return {
            "protocol": p.PROTOCOL_VERSION,
            "sample_rate": self.engine.sample_rate,
            "channels": 1,
            "format": "s16le",
            "model": self.engine.model_id,
            "max_text_chars": self.limits.max_text_chars,
            "max_inflight": self.limits.max_inflight,
        }

    def submit(self, req: Request) -> None:
        self._jobs.put(req)

    def track(self, conn: Connection, alive: bool) -> None:
        with self._conns_lock:
            if alive:
                self._conns.add(conn)
            else:
                self._conns.discard(conn)

    # -- inference worker --------------------------------------------------

    def _run_worker(self) -> None:
        # MLX default streams are thread-local, so the model must be loaded
        # and run on this same thread.
        try:
            self.engine = self.engine_factory()
            if self.warmup:
                self.engine.warmup()
        except BaseException as exc:
            log.exception("engine failed to start")
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()

        while True:
            item = self._jobs.get()
            if item is _STOP:
                return
            self._render(item)  # type: ignore[arg-type]

    def _render(self, req: Request) -> None:
        conn, rid, spec = req.conn, req.request_id, req.spec
        engine = self.engine
        assert engine is not None
        cancelled = p.error_frame(rid, p.ErrorCode.CANCELLED, "request cancelled")
        if req.cancelled.is_set():
            conn.finish(req, cancelled)
            return
        try:
            conn.send(p.start_frame(rid, engine.sample_rate))
            chunks = engine.stream_pcm(spec.text, voice=spec.voice, speed=spec.speed)
            try:
                for pcm in chunks:
                    if req.cancelled.is_set():
                        break
                    conn.send(p.audio_frame(rid, pcm))
            finally:
                chunks.close()
            conn.finish(req, cancelled if req.cancelled.is_set() else p.end_frame(rid))
        except Exception as exc:
            log.exception("synthesis failed for request %d", rid)
            conn.finish(req, p.error_frame(rid, p.ErrorCode.ENGINE_ERROR, f"{type(exc).__name__}: {exc}"))


def _remove_stale_socket(path: str) -> None:
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise FileExistsError(f"{path} exists and is not a socket")
    os.unlink(path)
