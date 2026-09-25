"""Python client for `speak-serve`. Stdlib only: no numpy, no model code.

    with SpeakClient() as client:
        for pcm in client.stream("Hello there.", voice="af_heart"):
            player.write(pcm)  # s16le mono at client.sample_rate
"""

from __future__ import annotations

import itertools
import queue
import socket
import threading
from typing import Dict, Iterator, Optional, Tuple, Union

from . import protocol as p
from .protocol import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_VOICE

# Server audio frames are one synthesized chunk each; allow long sentences.
MAX_AUDIO_PAYLOAD = 64 << 20

_DEAD = object()


class SpeakError(Exception):
    """The server rejected or failed a request, or the connection dropped.

    `code` is one of the server's error codes, or a client-side one:
    `connection_closed` or `timeout`.
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Response:
    """Handle for one in-flight request. Consume it with chunks()."""

    def __init__(self, client: "SpeakClient", request_id: int):
        self._client = client
        self.request_id = request_id
        self.start: Optional[dict] = None
        self.done = False
        self._frames: "queue.Queue[object]" = queue.Queue()

    def chunks(self, timeout: float | None = None) -> Iterator[bytes]:
        """Yield AUDIO payloads until END. Raises SpeakError on ERROR or a dead connection.

        `timeout` bounds the wait for each frame, not the whole request; when it
        expires the request is cancelled and SpeakError('timeout') is raised.
        """
        while not self.done:
            try:
                item = self._frames.get(timeout=timeout)
            except queue.Empty:
                self.done = True
                self._client._abandon(self)
                raise SpeakError("timeout", f"no frame from the server within {timeout}s") from None
            if item is _DEAD:
                self.done = True
                raise SpeakError(self._client.dead_code, self._client.dead_reason or "connection closed")
            frame: p.Frame = item  # type: ignore[assignment]
            if frame.type == p.FrameType.START:
                self.start = p.parse_json(frame)
            elif frame.type == p.FrameType.AUDIO:
                yield frame.payload
            elif frame.type == p.FrameType.END:
                self.done = True
            elif frame.type == p.FrameType.ERROR:
                self.done = True
                body = p.parse_json(frame)
                raise SpeakError(str(body.get("code", "unknown")), str(body.get("message", "")))

    def cancel(self) -> None:
        """Ask the server to stop. chunks() then ends with SpeakError('cancelled')
        unless the request had already finished."""
        self._client._send(p.cancel_frame(self.request_id))


class SpeakClient:
    """One connection to a speak server. Thread-safe; requests may be pipelined."""

    def __init__(
        self,
        address: Union[Tuple[str, int], str] = (DEFAULT_HOST, DEFAULT_PORT),
        connect_timeout: float = 5.0,
    ):
        if isinstance(address, str):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(connect_timeout)
            try:
                sock.connect(address)
            except BaseException:
                sock.close()
                raise
        else:
            sock = socket.create_connection(address, timeout=connect_timeout)
        self._sock = sock
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._pending: Dict[int, Response] = {}
        self._ids = itertools.count(1)
        self.dead_code = "connection_closed"
        self.dead_reason: Optional[str] = None
        try:
            hello = p.read_frame(self._read_exact, MAX_AUDIO_PAYLOAD)
            if hello is None or hello.type != p.FrameType.HELLO:
                raise SpeakError("protocol_error", "server did not send HELLO")
            self.hello = p.parse_json(hello)
        except BaseException:
            sock.close()
            raise
        self.sample_rate = int(self.hello["sample_rate"])
        sock.settimeout(None)
        self._reader = threading.Thread(target=self._read_loop, name="speak-client-reader", daemon=True)
        self._reader.start()

    # -- public API --------------------------------------------------------

    def submit(self, text: str, voice: str = DEFAULT_VOICE, speed: float | None = None) -> Response:
        """Send a request without waiting; read its audio from the returned Response."""
        with self._lock:
            if self.dead_reason is not None:
                raise SpeakError(self.dead_code, self.dead_reason)
            request_id = self._next_id()
            response = Response(self, request_id)
            self._pending[request_id] = response
        try:
            self._send(p.request_frame(request_id, text, voice, speed))
        except OSError as exc:
            with self._lock:
                self._pending.pop(request_id, None)
            raise SpeakError("connection_closed", str(exc)) from exc
        return response

    def stream(
        self, text: str, voice: str = DEFAULT_VOICE, speed: float | None = None, timeout: float | None = None
    ) -> Iterator[bytes]:
        """Yield s16le PCM chunks as they're synthesized. Stopping early cancels the request.

        `timeout` bounds the wait for each chunk; see Response.chunks().
        """
        response = self.submit(text, voice=voice, speed=speed)
        try:
            yield from response.chunks(timeout)
        finally:
            if not response.done:
                self._abandon(response)

    def synthesize(
        self, text: str, voice: str = DEFAULT_VOICE, speed: float | None = None, timeout: float | None = None
    ) -> Tuple[bytes, int]:
        """Return (s16le PCM, sample_rate) for the whole text."""
        response = self.submit(text, voice=voice, speed=speed)
        pcm = b"".join(response.chunks(timeout))
        rate = int(response.start["sample_rate"]) if response.start else self.sample_rate
        return pcm, rate

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        if self._reader is not threading.current_thread():
            self._reader.join()

    def __enter__(self) -> "SpeakClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    def _next_id(self) -> int:
        while True:
            request_id = next(self._ids) % 0x1_0000_0000
            if request_id and request_id not in self._pending:
                return request_id

    def _send(self, frame: p.Frame) -> None:
        with self._send_lock:
            self._sock.sendall(p.encode(frame))

    def _abandon(self, response: Response) -> None:
        """Cancel and stop routing frames to a response nobody will read."""
        with self._lock:
            if self._pending.get(response.request_id) is not response:
                return  # already finished, or the connection is gone
            del self._pending[response.request_id]
        try:
            response.cancel()
        except OSError:
            pass

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining:
            data = self._sock.recv(remaining)
            if not data:
                break
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _read_loop(self) -> None:
        code, reason = "connection_closed", "connection closed by server"
        try:
            while True:
                frame = p.read_frame(self._read_exact, MAX_AUDIO_PAYLOAD)
                if frame is None:
                    break
                if frame.request_id == 0 and frame.type == p.FrameType.ERROR:
                    # A connection-level error; the server closes the socket next.
                    body = p.parse_json(frame)
                    code = str(body.get("code", "protocol_error"))
                    reason = str(body.get("message", ""))
                    continue
                with self._lock:
                    response = self._pending.get(frame.request_id)
                    if frame.type in (p.FrameType.END, p.FrameType.ERROR):
                        self._pending.pop(frame.request_id, None)
                if response is not None:
                    response._frames.put(frame)
        except (OSError, p.ProtocolError) as exc:
            if code == "connection_closed":
                reason = f"connection lost: {exc}"
        with self._lock:
            self.dead_code = code
            self.dead_reason = reason
            pending = list(self._pending.values())
            self._pending.clear()
        for response in pending:
            response._frames.put(_DEAD)
