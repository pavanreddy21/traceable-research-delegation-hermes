#!/usr/bin/env bash
# Long-term memory demo: the second, related question recalls the first run's verified sources as hints.
set -euo pipefail
cd "$(dirname "$0")/.."
export TRACE_MEMORY_DIR="${TRACE_MEMORY_DIR:-results/memory_demo/memory}"
mkdir -p results/memory_demo
.venv/bin/python -u run.py --out results/memory_demo/run1 \
  "Does self-testing (retrieval practice) improve long-term retention more than rereading?" > /dev/null
.venv/bin/python -u run.py --out results/memory_demo/run2 \
  "Is practice testing better than restudying for students' delayed exam performance?" > /dev/null
python3 - <<'PY'
import json
rec = json.load(open("results/memory_demo/run2/evidence.json"))
mem = [t for t in rec["trace"] if t["event"] == "memory"]
print("memory recall events in run 2:", mem)
PY
