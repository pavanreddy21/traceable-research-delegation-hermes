"""Write results/RESULTS.md (and the README results section) from the saved results/*.json files.

    python experiments/summarize.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


def pct(x):
    return f"{100 * float(x):.0f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build():
    e1, e2, e4 = load("e1_checker_models.json"), load("e2_retrievers.json"), load("e4_safety_scope.json")
    e3 = {c: load(f"e3_eval_{c}.json").get("summary", {}) for c in ("full", "no_counter")}
    md = []
    if e1:
        md += ["### E1: claim-checker model (32 hand-labeled claims, `eval/claims_benchmark.jsonl`)", "",
               table(["Model", "Accuracy", "True claims accepted", "False claims accepted", "Mean request time", "Tokens"],
                     [[m, pct(r["accuracy"]), pct(r["true_accept_rate"]), pct(r["false_accept_rate"]),
                       f"{r['mean_request_seconds']} s",
                       f"{(r['prompt_tokens'] + r['completion_tokens']) / 1000:.0f}k"] for m, r in e1.items()]), ""]
    if e2:
        md += ["### E2: retriever (same claims vs. 51 passages)", "",
               table(["Retriever", "hit@1", "hit@3", "MRR"],
                     [[k, pct(r["hit@1"]), pct(r["hit@3"]), f"{r['mrr']:.3f}"] for k, r in e2.items()]), ""]
    if any(e3.values()):
        keys = [("Findings that passed checking", "checker_pass_rate", pct),
                ("Agent quotes found verbatim in source", "quote_found_rate", pct),
                ("Cited DOIs verified (Crossref / DataCite)", "citation_valid_rate", pct),
                ("Writer sentences citing verified evidence (before gate)", "writer_raw_support", pct),
                ("Sentences removed by citation gate (total)", "sentences_dropped_by_gate", str),
                ("Subquestions with verified support", "subquestion_coverage", pct),
                ("Briefs citing counter-evidence", "counterevidence_in_report", str),
                ("Briefs flagged for human review", "needs_human_review", str),
                ("Verified findings / question", "verified", lambda v: f"{v:.1f}"),
                ("LLM calls / question", "llm_calls", lambda v: f"{v:.1f}"),
                ("LLM request time / question (s, summed)", "llm_seconds", lambda v: f"{v:.0f}")]
        rows = [[label] + ["-" if e3[c].get(k) is None else (e3[c][k] if isinstance(e3[c][k], str) else f(e3[c][k]))
                           for c in ("full", "no_counter")] for label, k, f in keys]
        md += ["### E3: end to end on 8 questions (`eval/questions.jsonl`), full system vs. no counterexample agent", "",
               table(["Mean over questions", "Full", "No counterexample"], rows), "",
               "Every brief and evidence record: `results/runs/<config>/<question>/`.", ""]
    if e4:
        rows = []
        for layer in ("rules_only", "rules_plus_llm"):
            cases = e4.get(layer, {}).get("cases", [])
            miss = [c["id"] for c in cases if c["got"] != c["expect"]]
            rows.append([layer.replace("_", " "), pct(e4[layer]["accuracy"]), ", ".join(miss) or "none"])
        md += ["### E4: scope gate (10 seeded cases, `eval/safety_cases.jsonl`)", "",
               table(["Layer", "Accuracy", "Missed"], rows), ""]
    text = "\n".join(md) or "_Run `./experiments/run_all.sh` to produce results._"
    (RES / "RESULTS.md").write_text("# Results\n\n" + text + "\n")
    readme = ROOT / "README.md"
    r = readme.read_text()
    block = "<!-- results:start -->\n" + text + "\n<!-- results:end -->"
    if "%%RESULTS%%" in r:
        r = r.replace("%%RESULTS%%", block)
    else:
        r = re.sub(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, r, flags=re.S)
    readme.write_text(r)
    print("results/RESULTS.md and README updated")


if __name__ == "__main__":
    build()
