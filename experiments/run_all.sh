#!/usr/bin/env bash
# Runs every experiment in one queue so a rate-limited endpoint is never hit by two processes at once.
# With TRACE_MODE=replay this reproduces all saved numbers without a key.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
$PY experiments/e1_e2_components.py e2
$PY eval/run_eval.py --config full
$PY experiments/e1_e2_components.py e1
$PY eval/run_eval.py --safety
$PY eval/run_eval.py --config no_counter
