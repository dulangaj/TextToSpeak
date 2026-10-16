#!/usr/bin/env bash
# Runs the repo's texttospeak command from whichever venv has it.
set -euo pipefail
here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
for c in "$repo"/.venv*/bin/texttospeak "$(command -v texttospeak || true)"; do
  [ -n "$c" ] && [ -x "$c" ] && exec "$c" "$@"
done
echo "texttospeak: not installed. Run: cd \"$repo\" && python3 -m venv .venv && .venv/bin/pip install -e .   (Python 3.10+, Apple Silicon)" >&2
exit 127
