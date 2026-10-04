from __future__ import annotations

import re
import threading
import time

import numpy as np
import pytest

from speakctl.engine import DEFAULT_SPLIT_PATTERN, TTSEngine
from speakctl.server import SpeakServer


class FakeBackend:
    """Deterministic stand-in for the MLX model: one sine chunk per sentence."""

    def __init__(self, sample_rate=24000, chunk_seconds=0.05, fail_on=None, delay=0.0):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.fail_on = fail_on
        self.delay = delay
        self.calls = []
        self.chunks_yielded = 0
        self.lock = threading.Lock()

    def sentences(self, text):
        return [s for s in re.split(DEFAULT_SPLIT_PATTERN, text) if s and s.strip()]

    def chunk(self, index):
        n = int(self.sample_rate * self.chunk_seconds)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        freq = 220.0 * (index + 1)
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def render(self, text, voice, speed):
        with self.lock:
            self.calls.append((text, voice, speed))
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError(f"boom on {self.fail_on}")
        for index, _ in enumerate(self.sentences(text)):
            if self.delay:
                time.sleep(self.delay)
            with self.lock:
                self.chunks_yielded += 1
            yield self.chunk(index)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def engine(backend):
    return TTSEngine(model_id="fake-model", backend=backend)


def run_server(address, engine, **kwargs):
    server = SpeakServer(address, engine_factory=lambda: engine, **kwargs)
    server.start(timeout=10)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(5)


@pytest.fixture
def server_factory(engine):
    started = []

    def make(address=("127.0.0.1", 0), engine=engine, **kwargs):
        server, thread = run_server(address, engine, **kwargs)
        started.append((server, thread))
        return server

    yield make
    for server, thread in started:
        stop_server(server, thread)


@pytest.fixture
def server(server_factory):
    """Address of a running server with default limits and warm-up disabled."""
    return server_factory(warmup=False).server_address
