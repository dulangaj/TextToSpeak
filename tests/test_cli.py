import pytest

from speakctl.cli import _parse_job, serve_main


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
