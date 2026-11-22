"""Export a deterministic sample of verified findings for manual review (results/spot_check.csv).

A reviewer reads each claim next to the passage the checker relied on and marks it
supported / partial / not_supported in the review column.
    python experiments/spot_check.py [n=20]
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = []
for ev in sorted((ROOT / "results" / "runs" / "full").glob("*/evidence.json")):
    rec = json.loads(ev.read_text())
    for f in rec["findings"]:
        if f["check"].get("verdict") in ("supported", "partial"):
            rows.append({"question": ev.parent.name, "id": f["id"], "stance": f["stance"],
                         "checker_verdict": f["check"]["verdict"], "claim": f["claim"],
                         "passage": f["check"].get("passage", ""), "manual_review": "", "note": ""})
step = max(1, len(rows) // n)
sample = rows[::step][:n]
out = ROOT / "results" / "spot_check.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(sample[0]))
    w.writeheader()
    w.writerows(sample)
print(f"{len(sample)} of {len(rows)} verified findings -> {out}")
