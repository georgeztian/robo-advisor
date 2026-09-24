#!/usr/bin/env bash
# Setup (macOS / Linux): creates a private Python environment in .venv and installs the
# robo-advisor with Yahoo Finance support and the test tools. Safe to re-run (updates).
# You rarely need to call this yourself: ./ra runs it automatically whenever the environment
# is missing, incomplete, or older than pyproject.toml's dependency list.
#
#   ./setup.sh            install / update
#   ./setup.sh --fresh    delete .venv and reinstall from scratch
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--fresh" ]]; then
  echo "Removing existing .venv ..."
  rm -rf .venv
fi

# 1. find a Python >= 3.10
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [[ -z "$PY" ]]; then
  echo "ERROR: Python 3.10 or newer was not found."
  echo "Install it from https://www.python.org/downloads/ and run ./setup.sh again."
  exit 1
fi
echo "Using $("$PY" --version) ($(command -v "$PY"))"

# 2. create the private environment
if [[ ! -x .venv/bin/python ]]; then
  echo "Creating private Python environment in .venv ..."
  "$PY" -m venv .venv
fi

# 3. install the app and its libraries into it
echo "Installing robo-advisor and its libraries (this can take a few minutes) ..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e ".[yahoo,dev]"

# 4. smoke check, then record which dependency list was installed (ra compares this stamp)
.venv/bin/robo-advisor graph >/dev/null
.venv/bin/python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  pyproject.toml > .venv/.install-stamp
echo
echo "Setup complete. The environment is used automatically by ./ra - no activation needed:"
echo "  ./ra data                                                    # download + check real prices"
echo "  ./ra run --interactive --out clients/<name>                   # enter a client's answers"
echo "  ./ra run --profile clients/<name>/<name>_profile.json --as-of today --out clients/<name>"
