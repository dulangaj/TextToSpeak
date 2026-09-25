import errno
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import time

import pytest

from speakctl.client import SpeakClient, SpeakError
from speakctl.engine import TTSEngine
from speakctl.server import Limits, SpeakServer

from conftest import FakeBackend, stop_server

TEXT = "First sentence. Second one! Third?"


def long_text(n):
    return " ".join(f"Sentence number {i}." for i in range(n))


class Raw:
    """Speaks the wire protocol with nothing but socket + struct."""

    def __init__(self, address):
        self.sock = socket.create_connection(address, timeout=5)

    def recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            data = self.sock.recv(n - len(buf))
            if not data:
                return buf
            buf += data
        return buf

    def read(self):
        header = self.recv_exact(9)
        if not header:
            return None
        assert len(header) == 9
        ftype, rid, length = struct.unpack("!BII", header)
        return ftype, rid, self.recv_exact(length)

    def send(self, ftype, rid, payload=b""):
        self.sock.sendall(struct.pack("!BII", ftype, rid, len(payload)) + payload)

    def request(self, rid, text, voice="af_heart"):
        self.send(0x01, rid, json.dumps({"text": text, "voice": voice}).encode())

    def until_terminal(self, rid):
        frames = []
        while True:
            frame = self.read()
            assert frame is not None, "connection closed early"
            if frame[1] == rid:
                frames.append(frame)
                if frame[0] in (0x13, 0x14):
                    return frames

    def close(self):
        self.sock.close()


def test_hello(server, engine):
    with SpeakClient(server) as client:
        assert client.hello == {
            "protocol": 1,
            "sample_rate": 24000,
            "channels": 1,
            "format": "s16le",
            "model": "fake-model",
            "max_text_chars": Limits.max_text_chars,
            "max_inflight": Limits.max_inflight,
        }
        assert client.sample_rate == 24000


def test_single_request_matches_engine(server, engine):
    with SpeakClient(server) as client:
        response = client.submit(TEXT, voice="bf_emma", speed=1.5)
        chunks = list(response.chunks())
        assert response.start == {"sample_rate": 24000, "channels": 1, "format": "s16le"}
    assert len(chunks) == 3
    assert b"".join(chunks) == engine.synthesize_pcm(TEXT)


def test_backend_receives_voice_and_speed(server, backend):
    with SpeakClient(server) as client:
        client.synthesize("Hi.", voice="bf_emma", speed=1.5)
        client.synthesize("Hi.")
    assert backend.calls == [("Hi.", "bf_emma", 1.5), ("Hi.", "af_heart", 1.0)]


def test_stream_and_synthesize(server, engine):
    with SpeakClient(server) as client:
        assert b"".join(client.stream(TEXT)) == engine.synthesize_pcm(TEXT)
        pcm, rate = client.synthesize(TEXT)
    assert pcm == engine.synthesize_pcm(TEXT)
    assert rate == 24000


def test_concurrent_clients(server_factory):
    engine = TTSEngine(model_id="fake-model", backend=FakeBackend(delay=0.01))
    address = server_factory(engine=engine, warmup=False).server_address
    texts = [long_text(5), "Alpha. Beta.", long_text(3)]
    results = {}

    def run(i):
        with SpeakClient(address) as client:
            results[i] = client.synthesize(texts[i])[0]

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(texts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    for i, text in enumerate(texts):
        assert results[i] == engine.synthesize_pcm(text)


def test_pipelined_requests_complete_in_order(server, engine):
    raw = Raw(server)
    assert raw.read()[0] == 0x10
    texts = {1: "One. Two.", 2: "Three.", 3: "Four. Five. Six."}
    for rid, text in texts.items():
        raw.request(rid, text)
    seen = []
    audio = {rid: b"" for rid in texts}
    while len([f for f in seen if f[0] == 0x13]) < 3:
        ftype, rid, payload = raw.read()
        seen.append((ftype, rid))
        if ftype == 0x12:
            audio[rid] += payload
    raw.close()
    starts_and_ends = [f for f in seen if f[0] in (0x11, 0x13)]
    assert starts_and_ends == [(0x11, 1), (0x13, 1), (0x11, 2), (0x13, 2), (0x11, 3), (0x13, 3)]
    for rid, text in texts.items():
        assert audio[rid] == engine.synthesize_pcm(text)


def test_pipelined_via_client(server, engine):
    with SpeakClient(server) as client:
        responses = [client.submit(t) for t in ("A.", "B. C.", "D.")]
        # Read out of order; frames are buffered per request.
        results = [b"".join(r.chunks()) for r in reversed(responses)]
    assert results == [engine.synthesize_pcm(t) for t in ("D.", "B. C.", "A.")]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "x" * (Limits.max_text_chars + 1)},
        {"text": "Hi.", "voice": "../etc/passwd"},
        {"text": "Hi.", "voice": ""},
        {"text": "Hi.", "voice": "a" * 65},
        {"text": "Hi.", "speed": 0.1},
        {"text": "Hi.", "speed": 5.0},
        {"text": "Hi.", "speed": float("nan")},
    ],
)
def test_validation_errors(server, backend, kwargs):
    with SpeakClient(server) as client:
        with pytest.raises(SpeakError) as info:
            client.synthesize(**kwargs)
        assert info.value.code == "bad_request"
        # The connection stays usable.
        client.synthesize("Fine.")
    assert [c[0] for c in backend.calls] == ["Fine."]


def test_request_id_zero_rejected(server, backend):
    raw = Raw(server)
    raw.read()
    raw.request(0, "Hi.")
    ftype, rid, payload = raw.read()
    assert (ftype, rid) == (0x14, 0)
    assert json.loads(payload)["code"] == "bad_request"
    raw.close()
    assert backend.calls == []


def test_malformed_request_json_is_bad_request(server, backend):
    raw = Raw(server)
    raw.read()
    raw.send(0x01, 4, b'{"text": 1}')
    ftype, rid, payload = raw.read()
    assert (ftype, rid, json.loads(payload)["code"]) == (0x14, 4, "bad_request")
    raw.request(5, "Still works.")
    assert raw.until_terminal(5)[-1][0] == 0x13
    raw.close()


def test_duplicate_request_id_rejected(server_factory):
    backend = FakeBackend(delay=0.05)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    raw = Raw(address)
    raw.read()
    raw.request(7, long_text(4))
    raw.request(7, "Again.")
    # The duplicate is rejected (possibly after the original's START) and the original completes.
    frames = raw.until_terminal(7) + raw.until_terminal(7)
    errors = [f for f in frames if f[0] == 0x14]
    assert len(errors) == 1 and json.loads(errors[0][2])["code"] == "bad_request"
    rest = [f[0] for f in frames if f[0] != 0x14]
    assert rest == [0x11, 0x12, 0x12, 0x12, 0x12, 0x13]
    # Once finished, the id may be reused.
    raw.request(7, "Reuse.")
    assert raw.until_terminal(7)[-1][0] == 0x13
    raw.close()
    assert [c[0] for c in backend.calls] == [long_text(4), "Reuse."]


def test_inflight_cap(server_factory):
    backend = FakeBackend(delay=0.05)
    address = server_factory(
        engine=TTSEngine(backend=backend), warmup=False, limits=Limits(max_inflight=2)
    ).server_address
    with SpeakClient(address) as client:
        assert client.hello["max_inflight"] == 2
        a = client.submit(long_text(3))
        b = client.submit("B.")
        c = client.submit("C.")
        with pytest.raises(SpeakError) as info:
            list(c.chunks())
        assert info.value.code == "too_many_requests"
        list(a.chunks())
        list(b.chunks())
        # Slots free up once requests finish.
        client.synthesize("D.")
    assert [x[0] for x in backend.calls] == [long_text(3), "B.", "D."]


def test_engine_error_then_recovers(server_factory):
    backend = FakeBackend(fail_on="explode")
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    with SpeakClient(address) as client:
        with pytest.raises(SpeakError) as info:
            client.synthesize("Please explode.")
        assert info.value.code == "engine_error"
        assert "RuntimeError" in info.value.message
        assert client.synthesize("Calm.")[0]


def test_cancel_while_queued(server_factory):
    backend = FakeBackend(delay=0.05)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    with SpeakClient(address) as client:
        a = client.submit(long_text(4))
        b = client.submit("Never rendered.")
        b.cancel()
        with pytest.raises(SpeakError) as info:
            list(b.chunks())
        assert info.value.code == "cancelled"
        assert b.start is None
        list(a.chunks())
    assert [c[0] for c in backend.calls] == [long_text(4)]


def test_cancel_mid_stream(server_factory):
    backend = FakeBackend(delay=0.05)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    with SpeakClient(address) as client:
        response = client.submit(long_text(40))
        chunks = response.chunks()
        next(chunks)
        response.cancel()
        with pytest.raises(SpeakError) as info:
            list(chunks)
        assert info.value.code == "cancelled"
        assert client.synthesize("After.")[0]
    assert backend.chunks_yielded < 20


def test_stream_closed_early_cancels(server_factory):
    backend = FakeBackend(delay=0.05)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    with SpeakClient(address) as client:
        stream = client.stream(long_text(40))
        next(stream)
        stream.close()
        assert client.synthesize("After.")[0]
    assert backend.chunks_yielded < 20


def test_client_disconnect_mid_stream(server_factory):
    backend = FakeBackend(delay=0.05)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    first = SpeakClient(address)
    chunks = first.submit(long_text(40)).chunks()
    next(chunks)
    first.close()
    with SpeakClient(address) as second:
        assert second.synthesize("Still here.")[0]
    assert backend.chunks_yielded < 20


@pytest.mark.parametrize(
    "garbage",
    [
        b"\xff" * 9,  # unknown frame type
        struct.pack("!BII", 0x01, 1, 1 << 30),  # oversize payload
        struct.pack("!BII", 0x12, 1, 0),  # server-only frame type
    ],
)
def test_garbage_closes_connection(server, garbage):
    raw = Raw(server)
    raw.read()
    raw.sock.sendall(garbage)
    deadline = time.monotonic() + 5
    while True:
        frame = raw.read()
        if frame is None:
            break
        assert frame[:2] == (0x14, 0) and json.loads(frame[2])["code"] == "protocol_error"
        assert time.monotonic() < deadline
    raw.close()
    # The server itself is unaffected.
    with SpeakClient(server) as client:
        assert client.synthesize("Ok.")[0]


def test_unix_socket(server_factory, engine):
    # tmp_path can exceed the ~104-byte AF_UNIX path limit on macOS.
    directory = tempfile.mkdtemp(prefix="spk")
    path = os.path.join(directory, "s.sock")
    try:
        with open(path, "w"):
            pass
        with pytest.raises(FileExistsError):
            server_factory(address=path, warmup=False)
        os.unlink(path)

        server = server_factory(address=path, warmup=False)
        assert os.path.exists(path)
        with SpeakClient(path) as client:
            assert client.synthesize(TEXT)[0] == engine.synthesize_pcm(TEXT)
        server.shutdown()
        server.server_close()
        assert not os.path.exists(path)

        # A second server refuses a path a live server is listening on.
        first = server_factory(address=path, warmup=False)
        with pytest.raises(OSError) as info:
            SpeakServer(path, engine_factory=lambda: engine)
        assert info.value.errno == errno.EADDRINUSE
        with SpeakClient(path) as client:
            assert client.synthesize("Still first.")[0]
        first.shutdown()
        first.server_close()

        # A socket file left behind by a crashed server is replaced on startup.
        stale = socket.socket(socket.AF_UNIX)
        stale.bind(path)
        stale.close()
        server_factory(address=path, warmup=False)
        with SpeakClient(path) as client:
            assert client.synthesize("Again.")[0]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_raw_wire_contract(server, engine):
    """A non-Python client needs only: read 9 bytes, unpack !BII, read N."""
    raw = Raw(server)
    ftype, rid, payload = raw.read()
    assert (ftype, rid) == (0x10, 0)
    hello = json.loads(payload.decode("utf-8"))
    assert hello["format"] == "s16le" and hello["channels"] == 1

    raw.send(0x01, 42, json.dumps({"text": TEXT, "voice": "af_heart", "speed": 1.0}).encode())
    frames = raw.until_terminal(42)
    assert frames[0][0] == 0x11
    assert json.loads(frames[0][2]) == {"sample_rate": 24000, "channels": 1, "format": "s16le"}
    assert [f[0] for f in frames[1:-1]] == [0x12, 0x12, 0x12]
    assert frames[-1] == (0x13, 42, b"")
    assert b"".join(f[2] for f in frames[1:-1]) == engine.synthesize_pcm(TEXT)
    raw.close()


def test_startup_error_propagates():
    def broken():
        raise RuntimeError("no model")

    server = SpeakServer(("127.0.0.1", 0), engine_factory=broken)
    try:
        with pytest.raises(RuntimeError, match="no model"):
            server.start(timeout=5)
    finally:
        server.server_close()


def test_load_and_warmup_run_on_worker_thread():
    threads = []

    class Recording(FakeBackend):
        def render(self, text, voice, speed):
            threads.append(threading.current_thread().name)
            return super().render(text, voice, speed)

    backend = Recording()
    built_on = []

    def factory():
        built_on.append(threading.current_thread().name)
        return TTSEngine(backend=backend)

    srv = SpeakServer(("127.0.0.1", 0), engine_factory=factory)
    srv.start(timeout=5)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        with SpeakClient(srv.server_address) as client:
            client.synthesize("Hi.")
    finally:
        stop_server(srv, thread)
    assert built_on == ["speak-inference"]
    assert threads == ["speak-inference", "speak-inference"]


def test_client_sees_protocol_error_as_connection_error(server):
    client = SpeakClient(server)
    try:
        response = client.submit(long_text(3))
        client._sock.sendall(b"\xff" * 9)
        with pytest.raises(SpeakError) as info:
            list(response.chunks(timeout=5))
        assert info.value.code == "protocol_error"
        assert "unknown frame type" in info.value.message
        with pytest.raises(SpeakError):
            client.submit("Too late.")
    finally:
        client.close()


def small_rcvbuf_raw(address):
    """A raw client whose kernel receive buffer is tiny, so unread data backs up fast."""
    raw = Raw.__new__(Raw)
    raw.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    raw.sock.settimeout(5)
    raw.sock.connect(address)
    return raw


def test_slow_reader_does_not_block_others(server_factory):
    """The worker never waits on a client: others are served while one isn't reading."""
    backend = FakeBackend(chunk_seconds=1.0)  # ~48 KB per chunk, ~2.9 MB total
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    stalled = small_rcvbuf_raw(address)
    stalled.read()
    stalled.request(1, long_text(60))
    started = time.monotonic()
    with SpeakClient(address) as client:
        assert client.synthesize("Hi.", timeout=5)[0]
    assert time.monotonic() - started < 2
    # Well under outbox_bytes, so the stalled client's audio was buffered, not dropped.
    frames = stalled.until_terminal(1)
    assert frames[-1] == (0x13, 1, b"")
    assert len([f for f in frames if f[0] == 0x12]) == 60
    stalled.close()


def test_client_over_outbox_limit_is_dropped(server_factory):
    backend = FakeBackend(chunk_seconds=1.0)
    address = server_factory(
        engine=TTSEngine(backend=backend), warmup=False, limits=Limits(outbox_bytes=200_000)
    ).server_address
    stalled = small_rcvbuf_raw(address)
    stalled.read()
    stalled.request(1, long_text(200))  # ~9.6 MB, never read while rendering
    started = time.monotonic()
    with SpeakClient(address) as client:
        assert client.synthesize("Hi.", timeout=5)[0]
    assert time.monotonic() - started < 2
    # Draining what made it into the socket ends in EOF, never END.
    types = []
    while True:
        frame = stalled.read()
        if frame is None:
            break
        types.append(frame[0])
    assert 0x13 not in types and 0x14 not in types
    assert backend.chunks_yielded < 200
    stalled.close()


def test_client_close_unblocks_reader(server_factory):
    backend = FakeBackend(delay=0.2)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    client = SpeakClient(address)
    response = client.submit(long_text(40))
    errors = []

    def consume():
        try:
            list(response.chunks())
        except SpeakError as exc:
            errors.append(exc.code)

    thread = threading.Thread(target=consume)
    thread.start()
    time.sleep(0.1)
    client.close()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == ["connection_closed"]


def test_cancel_after_end_is_ignored(server, backend):
    raw = Raw(server)
    raw.read()
    raw.request(1, "Done.")
    assert raw.until_terminal(1)[-1][0] == 0x13
    raw.send(0x02, 1)
    raw.request(2, "Next.")
    frames = []
    while not frames or frames[-1][0] not in (0x13, 0x14):
        frames.append(raw.read())
    assert {f[1] for f in frames} == {2}
    assert frames[-1][0] == 0x13
    raw.close()


def test_half_close_cancels_inflight(server_factory):
    backend = FakeBackend(delay=0.05)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    raw = Raw(address)
    raw.read()
    raw.request(1, long_text(40))
    assert raw.read()[0] == 0x11
    raw.sock.shutdown(socket.SHUT_WR)
    types = []
    while True:
        frame = raw.read()
        if frame is None:
            break
        types.append(frame[0])
    raw.close()
    assert 0x13 not in types
    assert backend.chunks_yielded < 40


def test_chunks_timeout_cancels(server_factory):
    backend = FakeBackend(delay=0.5)
    address = server_factory(engine=TTSEngine(backend=backend), warmup=False).server_address
    with SpeakClient(address) as client:
        with pytest.raises(SpeakError) as info:
            client.synthesize(long_text(20), timeout=0.1)
        assert info.value.code == "timeout"
        assert client.synthesize("After.", timeout=5)[0]
    assert backend.chunks_yielded < 20
