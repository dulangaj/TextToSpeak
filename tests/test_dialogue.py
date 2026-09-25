import io
import wave

import pytest

from speakctl import dialogue
from speakctl.dialogue import DEFAULT_CAST, Turn, build_jobs, main, parse_line, resolve_cast
from speakctl.engine import TTSEngine
from speakctl.voices import KOKORO_VOICES
from conftest import FakeBackend


@pytest.fixture
def fake_engine(monkeypatch):
    backend = FakeBackend()
    made = []

    def factory(**kwargs):
        engine = TTSEngine(model_id=kwargs.get("model_id", "fake"), speed=kwargs.get("speed", 1.0), backend=backend)
        made.append(kwargs)
        return engine

    monkeypatch.setattr(dialogue, "TTSEngine", factory)
    backend.made = made
    return backend


def frames(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth()) == (1, 2)
        return wav.getnframes(), wav.getframerate()


def test_parse_line_alias():
    assert parse_line("voice2: Hi there", DEFAULT_CAST) == Turn("voice2", "am_michael", "Hi there")


def test_parse_line_raw_voice():
    assert parse_line("bf_lily:  Cheerio ", DEFAULT_CAST) == Turn("bf_lily", "bf_lily", "Cheerio")


def test_parse_line_bare_text_is_voice1():
    assert parse_line("  Just talking  ", DEFAULT_CAST) == Turn("voice1", "af_heart", "Just talking")


def test_parse_line_colon_after_several_words_is_bare_text():
    assert parse_line("Meet at 10:30", DEFAULT_CAST) == Turn("voice1", "af_heart", "Meet at 10:30")


def test_parse_line_bad_speaker():
    with pytest.raises(ValueError, match="'Bob'"):
        parse_line("Bob: hello", DEFAULT_CAST)


def test_resolve_cast_default():
    assert resolve_cast(None) == DEFAULT_CAST


def test_resolve_cast_override_merges():
    cast = resolve_cast("voice1=bf_alice, voice3=am_adam")
    assert cast == {"voice1": "bf_alice", "voice2": "am_michael", "voice3": "am_adam", "voice4": "bm_george"}
    assert DEFAULT_CAST["voice1"] == "af_heart"


@pytest.mark.parametrize("spec", ["voice9=af_heart", "voice1", "voice1=Not A Voice"])
def test_resolve_cast_rejects_bad_entries(spec):
    with pytest.raises(ValueError):
        resolve_cast(spec)


def test_parse_line_uses_cast_override():
    assert parse_line("voice1: hi", resolve_cast("voice1=bf_alice")).voice == "bf_alice"


def test_build_jobs_skips_empties():
    turns = build_jobs(["voice1: One.", "", "   ", "voice2:", "voice2:   ", "Two."], DEFAULT_CAST)
    assert [(t.speaker, t.text) for t in turns] == [("voice1", "One."), ("voice1", "Two.")]


def test_main_writes_conversation_and_split(tmp_path, fake_engine, capsys):
    out = tmp_path / "chat.wav"
    code = main(["voice1: Hello there. How are you?", "am_adam: Fine.", "Bare line.", "-o", str(out),
                 "--split", "--pause", "0.5", "--speed", "1.2"])
    assert code == 0

    chunk = int(fake_engine.sample_rate * fake_engine.chunk_seconds)
    pause = round(0.5 * fake_engine.sample_rate)
    assert frames(out) == ((2 + 1 + 1) * chunk + 2 * pause, 24000)

    split = [tmp_path / "001_voice1.wav", tmp_path / "002_am_adam.wav", tmp_path / "003_voice1.wav"]
    assert [frames(p)[0] for p in split] == [2 * chunk, chunk, chunk]
    assert capsys.readouterr().out.splitlines() == [str(out)] + [str(p) for p in split]

    assert [(voice, speed) for _, voice, speed in fake_engine.calls] == [
        ("af_heart", 1.2), ("am_adam", 1.2), ("af_heart", 1.2)
    ]
    assert fake_engine.made[0]["split_pattern"] is None


def test_main_split_outdir_and_cast(tmp_path, fake_engine, capsys):
    out = tmp_path / "chat.wav"
    code = main(["voice1: A.", "voice2: B.", "-o", str(out), "--split", "--outdir", str(tmp_path / "parts"),
                 "--cast", "voice2=bm_lewis", "--pause", "0"])
    assert code == 0
    chunk = int(fake_engine.sample_rate * fake_engine.chunk_seconds)
    assert frames(out)[0] == 2 * chunk
    assert (tmp_path / "parts" / "002_voice2.wav").exists()
    assert fake_engine.calls[1][1] == "bm_lewis"


def test_main_reads_stdin(tmp_path, fake_engine, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("voice1: One.\n\nvoice3: Two.\n"))
    out = tmp_path / "chat.wav"
    assert main(["-o", str(out)]) == 0
    assert [voice for _, voice, _ in fake_engine.calls] == ["af_heart", "bf_emma"]
    assert capsys.readouterr().out.splitlines() == [str(out)]


def test_main_bad_speaker_exits_2(tmp_path, fake_engine, capsys):
    with pytest.raises(SystemExit) as info:
        main(["voice1: hi", "Narrator: once upon a time", "-o", str(tmp_path / "x.wav")])
    assert info.value.code == 2
    assert "'Narrator'" in capsys.readouterr().err
    assert fake_engine.calls == []


def test_main_engine_failure_is_one_line(tmp_path, fake_engine, capsys):
    fake_engine.fail_on = "boom"
    code = main(["voice1: fine.", "voice2: boom", "-o", str(tmp_path / "x.wav")])
    assert code == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("texttospeak: error:")
    assert not (tmp_path / "x.wav").exists()


def test_list_voices(fake_engine, capsys):
    assert main(["--list-voices", "--cast", "voice4=zf_xiaoyi"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == ["voice1  af_heart", "voice2  am_michael", "voice3  bf_emma", "voice4  zf_xiaoyi"]
    assert lines[4] == ""
    assert tuple(lines[5:]) == KOKORO_VOICES
    assert len(KOKORO_VOICES) == 54
    assert fake_engine.made == []
