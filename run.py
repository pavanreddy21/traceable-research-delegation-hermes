"""Command line entry point.

    python run.py "Does self-testing improve memory more than rereading?"
    python run.py --remember "prefer meta-analyses"      # save a preference (explicit approval)
    TRACE_MODE=replay python run.py "<an evaluated question>"   # no key needed for recorded questions
"""
import argparse
import re
import sys

from traceable_research import config, memory
from traceable_research.coordinator import research


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--remember", help="save a stable, non-sensitive preference (only stored when you pass this)")
    p.add_argument("--no-counter", action="store_true", help="ablation: skip the counterexample searcher")
    p.add_argument("--no-tot", action="store_true", help="ablation: single plan, no follow-up search")
    p.add_argument("--no-memory", action="store_true", help="do not recall or save past runs")
    p.add_argument("--interactive", action="store_true", help="ask a clarifying question when the plan is ambiguous")
    p.add_argument("--out", help="output folder (default runs/<slug>)")
    a = p.parse_args()
    if a.remember:
        memory.remember_preference(a.remember)
        print(f"Saved preference: {a.remember}")
        if not a.question:
            return
    if not a.question:
        p.error("give a question")
    slug = re.sub(r"[^a-z0-9]+", "-", a.question.lower())[:60].strip("-")
    out = a.out or config.ROOT / "runs" / slug
    result = research(a.question, counterexample=not a.no_counter, tot=not a.no_tot,
                      use_memory=not a.no_memory, interactive=a.interactive, out_dir=out)
    if not a.no_memory and result["brief"]:
        memory.save_run(result["record"])
    print(result["markdown"])
    print(f"\nSaved: {out}/report.md and evidence.json", file=sys.stderr)


if __name__ == "__main__":
    main()
