import wave

import numpy as np
import pytest

from speakctl.engine import WARMUP_TEXT, Job, TTSEngine, float_to_s16le
from conftest import FakeBackend

TEXT = "First sentence. Second one! Third?"


def test_stream_pcm_one_chunk_per_backend_chunk(engine, backend):
    chunks = list(engine.stream_pcm(TEXT))
    assert len(chunks) == 3
    for index, pcm in enumerate(chunks):
        assert len(pcm) % 2 == 0
        expected = np.clip(backend.chunk(index), -1, 1) * 32767
        np.testing.assert_array_equal(np.frombuffer(pcm, dtype="<i2"), expected.astype("<i2"))


def test_float_to_s16le_values_and_clipping():
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -3.0], dtype=np.float32)
    out = np.frombuffer(float_to_s16le(samples), dtype="<i2")
    assert out.tolist() == [0, 16383, -16383, 32767, -32767, 32767, -32767]


def test_float_to_s16le_flattens_2d():
    assert len(float_to_s16le(np.zeros((1, 10), dtype=np.float32))) == 20


def test_synthesize_pcm_joins_stream(engine):
    assert engine.synthesize_pcm(TEXT) == b"".join(engine.stream_pcm(TEXT))


def test_skips_empty_chunks():
    class Sparse(FakeBackend):
        def render(self, text, voice, speed):
            yield np.zeros(0, dtype=np.float32)
            yield self.chunk(0)

    engine = TTSEngine(backend=Sparse())
    assert len(list(engine.stream_pcm("hi"))) == 1


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_text_raises(engine, text):
    with pytest.raises(ValueError):
        engine.synthesize_pcm(text)


def test_backend_yielding_nothing_raises():
    class Silent(FakeBackend):
        def render(self, text, voice, speed):
            return iter(())

    with pytest.raises(RuntimeError, match="no audio"):
        TTSEngine(backend=Silent()).synthesize_pcm("hello")


def test_synthesize_writes_wav(engine, backend, tmp_path):
    path = engine.synthesize(TEXT, tmp_path / "nested" / "out.wav")
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == backend.sample_rate
        assert wav.getnframes() == 3 * int(backend.sample_rate * backend.chunk_seconds)
        assert wav.readframes(wav.getnframes()) == engine.synthesize_pcm(TEXT)


def test_synthesize_uses_backend_sample_rate(tmp_path):
    engine = TTSEngine(backend=FakeBackend(sample_rate=16000))
    with wave.open(str(engine.synthesize("hi.", tmp_path / "a.wav")), "rb") as wav:
        assert wav.getframerate() == 16000


def test_synthesize_many_naming(engine, tmp_path):
    explicit = tmp_path / "custom.wav"
    jobs = [Job("One."), Job("Two.", voice="am_michael"), Job("Three.", output_path=explicit)]
    paths = engine.synthesize_many(jobs, outdir=tmp_path)
    assert paths == [tmp_path / "001_af_heart.wav", tmp_path / "002_am_michael.wav", explicit]
    assert all(p.exists() for p in paths)


def test_warmup_calls_backend_once(engine, backend):
    engine.warmup()
    assert [c[0] for c in backend.calls] == [WARMUP_TEXT]


def test_speed_default_and_override(backend):
    engine = TTSEngine(backend=backend, speed=1.3)
    engine.synthesize_pcm("a.", voice="bf_emma")
    engine.synthesize_pcm("b.", speed=0.8)
    assert backend.calls == [("a.", "bf_emma", 1.3), ("b.", "af_heart", 0.8)]


def test_default_backend_needs_mlx():
    try:
        import mlx  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("mlx is installed; this checks the lazy import on machines without it")
    with pytest.raises(ImportError) as info:
        TTSEngine()
    assert info.value.name.split(".")[0] in {"mlx", "mlx_audio"}
