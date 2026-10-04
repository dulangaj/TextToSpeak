import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run(code):
    return subprocess.run(
        [sys.executable, "-c", code], env={"PYTHONPATH": SRC}, capture_output=True, text=True, timeout=30
    )


def test_client_imports_without_numpy():
    result = run(
        "import sys; sys.modules['numpy'] = None\n"
        "import speakctl.client\n"
        "from speakctl import SpeakClient, SpeakError, DEFAULT_VOICE\n"
        "assert 'speakctl.server' not in sys.modules\n"
        "assert 'speakctl.engine' not in sys.modules\n"
        "try:\n"
        "    from speakctl import TTSEngine\n"
        "except ImportError:\n"
        "    print('engine-needs-numpy')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "engine-needs-numpy"


def test_engine_names_load_lazily():
    result = run("import speakctl, sys; speakctl.TTSEngine; speakctl.Job; print('speakctl.engine' in sys.modules)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"
