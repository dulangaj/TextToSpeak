import pytest

from speakctl.cli import _parse_job, serve_main
from speakctl.engine import TTSEngine


def test_serve_rejects_socket_with_port(capsys):
    with pytest.raises(SystemExit) as info:
        serve_main(["--socket", "/tmp/x.sock", "--port", "9000"])
    assert info.value.code == 2
    assert "--socket" in capsys.readouterr().err


def test_serve_rejects_socket_with_host(capsys):
    with pytest.raises(SystemExit) as info:
        serve_main(["--socket", "/tmp/x.sock", "--host", "127.0.0.1"])
    assert info.value.code == 2


def test_parse_job_with_voice():
    job = _parse_job("am_michael: hello there", "af_heart")
    assert (job.voice, job.text) == ("am_michael", " hello there")


def test_parse_job_bare_text_uses_default():
    job = _parse_job("hello there", "af_heart")
    assert (job.voice, job.text) == ("af_heart", "hello there")


def test_parse_job_empty_voice_keeps_whole_spec():
    job = _parse_job(": hello", "bf_emma")
    assert (job.voice, job.text) == ("bf_emma", ": hello")


def test_serve_rejects_unknown_log_level():
    with pytest.raises(SystemExit):
        serve_main(["--log-level", "chatty"])


def test_parse_job_colon_after_several_words_is_bare_text():
    job = _parse_job("Meet at 10:30", "af_heart")
    assert (job.voice, job.text) == ("af_heart", "Meet at 10:30")


def test_watch_survives_a_failing_line(tmp_path, monkeypatch, capsys):
    import io
    import sys

    from conftest import FakeBackend
    from speakctl import cli

    backend = FakeBackend(fail_on="bad")
    monkeypatch.setattr(cli, "TTSEngine", lambda **kwargs: TTSEngine(backend=backend))
    monkeypatch.setattr(sys, "argv", ["speak", "--watch", "-d", str(tmp_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO("bad line\nam_michael: good line\n"))
    assert cli.main() == 0
    out = capsys.readouterr()
    assert out.out.split() == [str(tmp_path / "002_am_michael.wav")]
    assert "line 1: boom on bad" in out.err
