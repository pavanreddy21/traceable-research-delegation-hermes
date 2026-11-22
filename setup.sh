#!/usr/bin/env bash
# One-step setup: virtual environment, dependencies, .env template, offline tests.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ required"'
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
TRACE_MODE=replay .venv/bin/python -m unittest discover -s tests
echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  TRACE_MODE=replay python eval/run_eval.py        # reproduce the saved results, no key needed"
echo "  python run.py \"your research question\"           # live run: add your Cloudflare keys to .env first"
