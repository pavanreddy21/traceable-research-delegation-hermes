"""End-to-end evaluation of the full system and its ablations (experiment E3).

    python eval/run_eval.py                      # full system on eval/questions.jsonl
    python eval/run_eval.py --config no_counter  # ablation: no counterexample searcher
    python eval/run_eval.py --safety             # scope gate on eval/safety_cases.jsonl (no LLM)
    TRACE_MODE=replay python eval/run_eval.py    # reproduce the saved numbers with no key, no network
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traceable_research import config  # noqa: E402
from traceable_research.coordinator import research  # noqa: E402
from traceable_research.guardrails import scope_check  # noqa: E402

CONFIGS = {"full": {}, "no_counter": {"counterexample": False}}  # run.py --no-tot exists for manual tests


def metrics(result):
    rec, brief, stats = result["record"], result["brief"] or {}, result["stats"]
    findings = rec.findings
    verified = rec.verified()
    text = " ".join([brief.get("answer", "")] + brief.get("key_findings", []) + brief.get("limits_and_conflicts", []))
    cited = set(re.findall(r"E\d+", text))
    cited_f = [f for f in verified if f.id in cited]
    cites_ok = [f.check.get("citation", {}).get("title_match", False) for f in cited_f]
    subqs = rec.plan.get("subquestions", [])
    covered = sum(any(f.subquestion.startswith(s["q"]) and f.stance == "supports" for f in verified) for s in subqs)
    return {
        "findings": len(findings),
        "verified": len(verified),
        "checker_pass_rate": round(len(verified) / len(findings), 3) if findings else 0,
        "quote_found_rate": round(sum(f.check.get("quote_found", False) for f in findings) / len(findings), 3)
        if findings else 0,
        "report_sentences": len([s for s in brief.get("key_findings", [])]) + len(brief.get("limits_and_conflicts", [])),
        "writer_raw_support": round(brief.get("raw_supported", 0) / brief["raw_sentences"], 3)
        if brief.get("raw_sentences") else 0,
        "sentences_dropped_by_gate": len(brief.get("dropped", [])),
        "citation_valid_rate": round(sum(cites_ok) / len(cites_ok), 3) if cites_ok else 0,
        "counterevidence_in_report": any(f.stance in ("challenges", "limits") for f in cited_f),
        "subquestion_coverage": round(covered / len(subqs), 3) if subqs else 0,
        "needs_human_review": rec.needs_human_review,
        "recoveries": sum(t["event"] == "recover" for t in rec.trace),
        "guardrail_events": sum(t["event"] == "guardrail" for t in rec.trace),
        "searches": len(rec.queries_tried),
        "rounds": 1 + sum(d["decision"] == "follow-up" for d in rec.decisions),
        **stats,
    }


def summarize(rows):
    keys = ["verified", "checker_pass_rate", "quote_found_rate", "writer_raw_support", "citation_valid_rate",
            "subquestion_coverage", "recoveries", "llm_calls", "prompt_tokens", "completion_tokens", "llm_seconds",
            "wall_seconds"]
    s = {k: round(statistics.mean(r[k] for r in rows), 3) for k in keys}
    s["counterevidence_in_report"] = f"{sum(r['counterevidence_in_report'] for r in rows)}/{len(rows)}"
    s["needs_human_review"] = f"{sum(r['needs_human_review'] for r in rows)}/{len(rows)}"
    s["sentences_dropped_by_gate"] = sum(r["sentences_dropped_by_gate"] for r in rows)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="full", choices=CONFIGS)
    p.add_argument("--only", help="comma-separated question ids")
    p.add_argument("--safety", action="store_true")
    a = p.parse_args()
    out = config.ROOT / "results"
    out.mkdir(exist_ok=True)
    if a.safety:
        cases = [json.loads(line) for line in open(config.ROOT / "eval" / "safety_cases.jsonl")]
        result = {}
        for layer, use_llm in (("rules_only", False), ("rules_plus_llm", True)):
            rows = [{**c, "got": scope_check(c["question"], use_llm=use_llm)["decision"]} for c in cases]
            for r in rows:
                print(f"{layer:15} {r['id']:4} expect={r['expect']:8} got={r['got']:8} "
                      f"{'OK' if r['got'] == r['expect'] else 'MISS'}")
            acc = sum(r["got"] == r["expect"] for r in rows) / len(rows)
            print(f"{layer}: scope gate accuracy {acc:.2f}")
            result[layer] = {"accuracy": round(acc, 3), "cases": rows}
        (out / "e4_safety_scope.json").write_text(json.dumps(result, indent=1))
        return
    qs = [json.loads(line) for line in open(config.ROOT / "eval" / "questions.jsonl")]
    if a.only:
        qs = [q for q in qs if q["id"] in a.only.split(",")]
    rows = []
    for q in qs:
        result = research(q["question"], use_memory=False, out_dir=out / "runs" / a.config / q["id"],
                          **CONFIGS[a.config])
        m = {"id": q["id"], **metrics(result)}
        rows.append(m)
        print(json.dumps(m))
    summary = summarize(rows)
    print("SUMMARY", json.dumps(summary))
    (out / f"e3_eval_{a.config}.json").write_text(json.dumps({"summary": summary, "questions": rows}, indent=1))


if __name__ == "__main__":
    main()
