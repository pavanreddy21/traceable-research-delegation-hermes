"""Long-term memory, kept deliberately small (design checkpoint 1.2):

- preferences.json: stable, non-sensitive preferences, written ONLY when the user passes --remember.
- runs.jsonl: what past tasks verified. Recalled as *hints* for new tasks; never trusted as current truth,
  every hinted source is re-fetched and re-checked.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import config
from .llm import embed

DIR = Path(config.os.environ.get("TRACE_MEMORY_DIR", config.ROOT / "memory"))


def remember_preference(text):
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / "preferences.json"
    prefs = json.loads(path.read_text()) if path.exists() else []
    if text not in prefs:
        prefs.append(text)
    path.write_text(json.dumps(prefs, indent=1))


def preferences():
    path = DIR / "preferences.json"
    return json.loads(path.read_text()) if path.exists() else []


def save_run(record):
    DIR.mkdir(parents=True, exist_ok=True)
    sources = [{"source_id": f.source_id, "title": record.sources[f.source_id]["title"]}
               for f in record.verified() if f.source_id in record.sources]
    row = {"question": record.question, "date": time.strftime("%Y-%m-%d"), "verified_sources": sources}
    with open(DIR / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def recall(question, k=2, threshold=0.6):
    """Return past runs whose question is semantically close to this one."""
    path = DIR / "runs.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["question"] != question and r["verified_sources"]]
    if not rows:
        return []
    vecs = np.array(embed([question] + [r["question"] for r in rows]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs[1:] @ vecs[0]
    order = np.argsort(-sims)[:k]
    return [rows[i] | {"similarity": round(float(sims[i]), 3)} for i in order if sims[i] >= threshold]
