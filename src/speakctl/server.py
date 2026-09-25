"""Long-lived TTS server: load the model once, stream PCM to many clients.

One inference worker thread owns the engine and renders requests in arrival
order. Each connection has a reader (the socketserver handler thread) and a
writer thread that drains a bounded outbox, so a slow client never blocks
another client's reads. See protocol.py for the wire format.
"""

from __future__ import annotations

import collections
import errno
import logging
import os
import queue
import re
import socket
import socketserver
import stat
import threading
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional, Set, Tuple, Union

from . import protocol as p
from .engine import TTSEngine

log = logging.getLogger(__name__)

VOICE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SPEED_MIN, SPEED_MAX = 0.5, 2.0
DEFAULT_HOST = p.DEFAULT_HOST
DEFAULT_PORT = p.DEFAULT_PORT

_STOP = object()


@dataclass
class Limits:
    max_text_chars: int = 10_000
    max_inflight: int = 32  # per connection: queued + rendering
    # Unsent bytes buffered per connection. 64 MiB is ~23 minutes of 24 kHz
    # s16le, so a client playing audio as it arrives never hits it; one that
    # stops reading is dropped rather than stalling the inference worker.
    outbox_bytes: int = 64 << 20
    close_timeout: float = 5.0  # how long a closing connection may take to flush


@dataclass
class Request:
    conn: "Connection"
    request_id: int
    spec: p.RequestSpec
    cancelled: threading.Event = field(default_factory=threading.Event)

    @property
    def stopped(self) -> bool:
        return self.cancelled.is_set() or self.conn.closed


class Connection:
    """One client socket: an outbox drained by a writer thread, plus in-flight requests.

    send() never blocks, so the shared inference worker can't be held up by
    one slow client. If a client falls `outbox_bytes` behind, it's dropped.
    """

    def __init__(self, sock: socket.socket, limits: Limits):
        self.sock = sock
        self.limits = limits
        self.inflight: Dict[int, Request] = {}
        self.lock = threading.Lock()  # guards inflight and closed
        self.closed = False
        self._outbox: Deque[p.Frame] = collections.deque()
        self._outbox_bytes = 0
        self._stopping = False
        self._cond = threading.Condition()  # guards the outbox fields above
        self._writer = threading.Thread(target=self._write_loop, name="speak-writer", daemon=True)
        self._writer.start()

    def send(self, frame: p.Frame) -> None:
        """Queue a frame for the client without blocking. A no-op once closed."""
        size = p.HEADER_SIZE + len(frame.payload)
        with self._cond:
            if self._stopping:
                return
            if self._outbox_bytes + size <= self.limits.outbox_bytes:
                self._outbox.append(frame)
                self._outbox_bytes += size
                self._cond.notify()
                return
        log.warning("client is more than %d bytes behind; dropping connection", self.limits.outbox_bytes)
        self.close(abort=True)

    def close(self, abort: bool = False) -> None:
        """Stop accepting frames and cancel in-flight work. Idempotent.

        Normally frames already queued are flushed before the socket is shut
        down; `abort=True` discards them and shuts it down immediately.
        """
        with self.lock:
            self.closed = True
            for req in self.inflight.values():
                req.cancelled.set()
        with self._cond:
            self._stopping = True
            if abort:
                self._outbox.clear()
                self._outbox_bytes = 0
            self._cond.notify()
        if abort:
            self._shutdown()

    def join(self) -> None:
        """Wait for queued frames to flush; force the socket down if the client won't read."""
        self._writer.join(self.limits.close_timeout)
        if self._writer.is_alive():
            self.close(abort=True)
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
            with self._cond:
                while not self._outbox and not self._stopping:
                    self._cond.wait()
                if not self._outbox:
                    break
                frame = self._outbox.popleft()
                self._outbox_bytes -= p.HEADER_SIZE + len(frame.payload)
            try:
                self._send_frame(frame)
            except OSError:
                self.close(abort=True)
                return
        self._shutdown()

    def _send_frame(self, frame: p.Frame) -> None:
        # Scatter-gather so large AUDIO payloads aren't copied to prepend the header.
        header = p.encode_header(frame)
        if not frame.payload:
            self.sock.sendall(header)
            return
        sent = self.sock.sendmsg([header, frame.payload])
        if sent < len(header):
            self.sock.sendall(header[sent:])
            sent = len(header)
        if sent - len(header) < len(frame.payload):
            self.sock.sendall(memoryview(frame.payload)[sent - len(header):])


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
        try:
            while not conn.closed:
                frame = p.read_frame(self._read_exact)
                if frame is None:
                    return
                if frame.type == p.FrameType.REQUEST:
                    self._on_request(conn, frame)
                elif frame.type == p.FrameType.CANCEL:
                    conn.cancel(frame.request_id)
                else:
                    raise p.ProtocolError(f"{frame.type.name} frames are server-to-client only")
        except p.ProtocolError as exc:
            log.info("protocol error from client: %s", exc)
            # id 0: the error is about the connection, which closes after this frame.
            conn.send(p.error_frame(0, p.ErrorCode.PROTOCOL_ERROR, str(exc)))

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
            if conn.closed:
                return
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
        self._unix_inode: Optional[int] = None
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
            self._unix_inode = os.stat(self.unix_path).st_ino
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
                # Only remove our own socket file, not one a newer server bound since.
                if os.stat(self.unix_path).st_ino == self._unix_inode:
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
        if req.stopped:
            conn.finish(req, cancelled)
            return
        try:
            conn.send(p.start_frame(rid, engine.sample_rate))
            chunks = engine.stream_pcm(spec.text, voice=spec.voice, speed=spec.speed)
            try:
                for pcm in chunks:
                    if req.stopped:
                        break
                    conn.send(p.audio_frame(rid, pcm))
            finally:
                chunks.close()
            conn.finish(req, cancelled if req.stopped else p.end_frame(rid))
        except Exception as exc:
            log.exception("synthesis failed for request %d", rid)
            conn.finish(req, p.error_frame(rid, p.ErrorCode.ENGINE_ERROR, f"{type(exc).__name__}: {exc}"))


def _remove_stale_socket(path: str) -> None:
    """Remove a socket file left by a server that's gone; refuse if one is still listening."""
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise FileExistsError(f"{path} exists and is not a socket")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(path)
    except ConnectionRefusedError:
        os.unlink(path)
        return
    except FileNotFoundError:
        return
    finally:
        probe.close()
    raise OSError(errno.EADDRINUSE, f"another server is listening on {path}")
