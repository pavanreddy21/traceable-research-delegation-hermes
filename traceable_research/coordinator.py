"""The coordinator: scope check -> Tree-of-Thought plan -> parallel role agents -> claim checks ->
coverage review (bounded ToT follow-ups) -> cited report with conflicts and gaps."""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, memory
from .agents import RoleAgent, verify_finding
from .evidence import EvidenceRecord
from .guardrails import BudgetExceeded, scope_check
from .llm import USAGE, chat, parse_json
from .report import render_markdown

PLAN = """You plan research for this question: "{question}"
{narrow}
Propose {n} DIFFERENT plans. Each plan breaks the question into 2 or 3 subquestions that can be researched
independently in scholarly literature, each with one short search query (4-8 words).
If the question is ambiguous in a way that would change the answer, set "ambiguity" to the clarifying
question you would ask, else null.
Reply with JSON only:
{{"ambiguity": null, "candidates": [{{"subquestions": [{{"q": "...", "query": "..."}}]}}]}}"""

SCORE = """Question: "{question}"
Score each candidate {what} from 0 to 10 on: covers what the question needs (0-4), each item is specific and
searchable (0-3), includes conditions or limits that could change the answer (0-3).
{candidates}
Reply with JSON only: {{"scores": [<one number per candidate, in order>], "why": "<one sentence>"}}"""

FOLLOWUP = """Question: "{question}"
These subquestions still lack verified evidence: {gaps}
Queries already tried: {tried}
Propose {n} different next actions. Each is a new search for one gap, worded differently from the tried
queries, or "accept_gap" if more search is unlikely to help.
Reply with JSON only: {{"actions": [{{"action": "search|accept_gap", "subquestion": "...", "query": "..."}}]}}"""

WRITE = """Write a short research brief that answers: "{question}"
{prefs}
Use ONLY the verified evidence below. Cite evidence ids in square brackets, like [E2], after every factual
sentence. Do not add facts that are not in the evidence. If sources disagree, say so and explain any
difference in population, task or setting.

Evidence:
{evidence}

Reply with JSON only:
{{"answer": "<2-4 sentences that directly answer, with citations>",
  "key_findings": ["<sentence with [E#]>", ...],
  "limits_and_conflicts": ["<sentence with [E#]>", ...],
  "open_questions": ["<what the evidence does not settle>", ...],
  "unresolved_conflict": <true if credible sources contradict each other and the evidence cannot explain why>}}"""


def _ask(prompt, max_tokens=900):
    reply = chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return parse_json(reply)


def _ask_retry(prompt, record, what, max_tokens=900):
    """One retry with the parse error fed back (recovering from a malformed reply)."""
    try:
        return _ask(prompt, max_tokens)
    except ValueError as err:
        record.log("coordinator", "recover", problem=f"unparseable {what}", detail=str(err))
        return _ask(prompt + f"\n\nYour previous reply was invalid ({err}). Reply with valid JSON only.", max_tokens)


def _score(question, items, what, record):
    listing = "\n".join(f"Candidate {i + 1}: {c}" for i, c in enumerate(items))
    try:
        j = _ask_retry(SCORE.format(question=question, what=what, candidates=listing), record, "scores", 300)
        scores = [float(s) for s in j.get("scores", [])][:len(items)]
        why = j.get("why", "")
    except (ValueError, TypeError):
        scores, why = [], "scorer failed"
    scores += [0.0] * (len(items) - len(scores))
    return scores, why


def plan(question, record, narrow_note="", tot=True):
    n = config.MAX_BRANCHES if tot else 1
    j = _ask_retry(PLAN.format(question=question, narrow=narrow_note, n=n), record, "plan")
    cands = [c for c in j.get("candidates", []) if c.get("subquestions")][:n]
    if not cands:
        cands = [{"subquestions": [{"q": question, "query": question[:80]}]}]
    if tot and len(cands) > 1:
        scores, why = _score(question, [[s["q"] for s in c["subquestions"]] for c in cands], "plan", record)
        best = max(range(len(cands)), key=lambda i: scores[i])
        record.decisions.append({"decision": "plan", "candidates": cands, "scores": scores,
                                 "chosen": best, "why": why})
    else:
        best = 0
        record.decisions.append({"decision": "plan", "candidates": cands, "chosen": 0, "why": "single plan (ToT off)"})
    chosen = cands[best]["subquestions"][:3]
    return {"ambiguity": j.get("ambiguity"), "subquestions": chosen}


def _run_agents(tasks, record, model):
    """Run role agents in parallel; add their findings in a fixed order so ids are deterministic."""
    def one(task):
        role, sub, hint = task
        agent = RoleAgent(role, record, model)
        try:
            return agent.run(sub, hint), agent.queries
        except BudgetExceeded as err:
            record.log(role, "budget", problem=str(err))
            return [], getattr(agent, "queries", [])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, tasks))
    for _, queries in results:
        record.queries_tried.extend(queries)
    new = []
    for found, _ in results:
        for f in found:
            dup = any(g.source_id == f["source_id"] and g.claim == f["claim"] for g in record.findings)
            if not dup:
                new.append(record.add_finding(**f))
    return new


def _check(findings, record, retriever):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: verify_finding(f, record, retriever=retriever), findings))


def _gaps(subqs, record, counterexample):
    verified = record.verified()
    gaps = [s for s in subqs if not any(f.subquestion.startswith(s["q"]) and f.stance == "supports"
                                        for f in verified)]
    if counterexample and not any(f.stance in ("challenges", "limits") for f in verified):
        gaps.append({"q": "credible counterevidence or limits", "query": "", "counter": True})
    return gaps


def research(question, *, counterexample=True, tot=True, retriever=None, model=None, use_memory=True,
             interactive=False, out_dir=None):
    USAGE.reset()
    t0 = time.time()
    record = EvidenceRecord(question)
    record.scope = scope_check(question, use_llm=True)
    if record.scope["decision"] == "decline":
        record.log("coordinator", "guardrail", problem=f"declined: {record.scope['reason']}")
        return _finish(record, None, t0, out_dir)
    narrow = ""
    if record.scope["decision"] == "narrow":
        narrow = ("This asks for individual advice. Plan ONLY general public research on the topic; the brief "
                  "will not give personal advice.")

    hints = ""
    if use_memory:
        past = memory.recall(question)
        if past:
            ids = sorted({s["source_id"] for r in past for s in r["verified_sources"]})[:6]
            hints = (f"\nHint from past tasks (unverified, re-fetch and re-check before use): sources {', '.join(ids)}"
                     " were relevant to similar questions.")
            record.log("coordinator", "memory", recalled=[r["question"] for r in past])

    record.plan = plan(question, record, narrow, tot)
    if record.plan.get("ambiguity"):
        if interactive:
            answer = input(f"Clarify: {record.plan['ambiguity']}\n> ").strip()
            if answer:
                question = f"{question} (clarified: {answer})"
                record.plan = plan(question, record, narrow, tot)
        else:
            record.gaps.append(f"Ambiguity not clarified (non-interactive run): {record.plan['ambiguity']}")
    subqs = record.plan["subquestions"]

    tasks = [("source_finder", f"{s['q']} (suggested query: {s['query']})", hints) for s in subqs]
    if counterexample:
        tasks.append(("counterexample", f"{question} (look for evidence against or limits)", ""))
    for rnd in range(config.MAX_ROUNDS):
        new = _run_agents(tasks, record, model)
        _check(new, record, retriever)
        gaps = _gaps(subqs, record, counterexample)
        record.log("coordinator", "review", round=rnd, verified=len(record.verified()), gaps=[g["q"] for g in gaps])
        if not gaps or rnd == config.MAX_ROUNDS - 1 or USAGE.calls >= config.MAX_LLM_CALLS:
            break
        if not tot:  # ablation: no follow-up search, report the gaps as they are
            break
        tasks = _followups(question, gaps, record, counterexample)
        if not tasks:
            break

    for g in _gaps(subqs, record, counterexample):
        record.gaps.append(f"No verified evidence for: {g['q']}")
    brief = _write(question, record)
    return _finish(record, brief, t0, out_dir)


def _followups(question, gaps, record, counterexample):
    j = _ask_retry(FOLLOWUP.format(question=question, gaps=[g["q"] for g in gaps], n=config.MAX_BRANCHES,
                                   tried=record.queries_tried[-8:]), record, "follow-ups", 500)
    acts = [a for a in j.get("actions", []) if isinstance(a, dict)][:config.MAX_BRANCHES]
    if not acts:
        return []
    scores, why = _score(question, [f"{a.get('action')}: {a.get('query', '')}" for a in acts], "next action", record)
    order = sorted(range(len(acts)), key=lambda i: -scores[i])[:config.KEEP_BRANCHES]
    record.decisions.append({"decision": "follow-up", "candidates": acts, "scores": scores, "kept": order, "why": why})
    tasks = []
    for i in order:
        a = acts[i]
        if a.get("action") != "search" or not a.get("query"):
            continue
        sub = a.get("subquestion") or gaps[0]["q"]
        counter = "counter" in sub.lower() or "limit" in sub.lower()
        role = "counterexample" if counter and counterexample else "source_finder"
        match = next((g["q"] for g in gaps if g["q"].lower() in sub.lower() or sub.lower() in g["q"].lower()), sub)
        tasks.append((role, f"{match} (suggested query: {a['query']})", ""))
    return tasks


def _write(question, record):
    verified = record.verified()
    if not verified:
        record.needs_human_review = True
        return {"answer": "The agents could not find verified evidence for this question within the task budget.",
                "key_findings": [], "limits_and_conflicts": [], "open_questions": [question],
                "unresolved_conflict": False, "dropped": [], "raw_sentences": 0, "raw_supported": 0}
    lines = []
    for f in verified:
        s = record.sources[f.source_id]
        lines.append(f"[{f.id}] ({f.stance}, {f.check['verdict']}) {f.claim}\n   source: {', '.join(s['authors'][:2])}"
                     f" ({s['year']}), \"{s['title']}\"\n   passage: \"{f.check['passage']}\"")
    prefs = memory.preferences()
    prefs_note = f"User preferences: {'; '.join(prefs)}" if prefs else ""
    try:
        brief = _ask_retry(WRITE.format(question=question, evidence="\n".join(lines), prefs=prefs_note),
                           record, "brief", 1200)
    except ValueError:
        brief = {"answer": "Report writer failed; see the verified evidence below.", "key_findings": [],
                 "limits_and_conflicts": [], "open_questions": [], "unresolved_conflict": True}
    return _gate(brief, record)


def _gate(brief, record):
    """Drop any sentence that cites nothing, or cites evidence that is missing or failed checking."""
    ok = {f.id for f in record.verified()}
    dropped, raw, raw_ok = [], 0, 0
    for key in ("answer", "key_findings", "limits_and_conflicts"):
        items = brief.get(key) or []
        single = isinstance(items, str)
        sents = re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", items) if single else [str(x) for x in items]
        kept = []
        for s in sents:
            if not s.strip():
                continue
            cites = re.findall(r"E\d+", s)
            raw += 1
            if cites and all(c in ok for c in cites):
                raw_ok += 1
                kept.append(s)
            else:
                dropped.append(s)
        brief[key] = " ".join(kept) if single else kept
    brief["dropped"], brief["raw_sentences"], brief["raw_supported"] = dropped, raw, raw_ok
    if dropped:
        record.log("coordinator", "guardrail", problem=f"dropped {len(dropped)} uncited or unverified sentences")
    contested = {f.stance for f in record.verified()} >= {"supports", "challenges"}
    if brief.get("unresolved_conflict") and contested:
        record.needs_human_review = True
        record.conflicts.append("Writer reported credible sources that contradict each other without explanation.")
    if any(record.sources[f.source_id].get("injection_flags") for f in record.verified()):
        record.needs_human_review = True
        record.conflicts.append("A cited source contained instruction-like text (removed); check it by hand.")
    return brief


def _finish(record, brief, t0, out_dir):
    stats = USAGE.as_dict() | {"wall_seconds": round(time.time() - t0, 1)}
    out = {"record": record, "brief": brief, "stats": stats}
    out["markdown"] = render_markdown(record, brief, stats)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        record.save(out_dir / "evidence.json")
        (out_dir / "report.md").write_text(out["markdown"])
    return out
