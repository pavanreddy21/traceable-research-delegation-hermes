"""Render the brief + evidence record as a Markdown report a reader can audit."""


def _cite(s):
    who = ", ".join(s.get("authors", [])[:2]) + (" et al." if len(s.get("authors", [])) > 2 else "")
    return f"{who} ({s.get('year')}). {s.get('title')}. {s.get('venue') or ''}".strip()


def render_markdown(record, brief, stats):
    out = [f"# Research brief", "", f"**Question:** {record.question}", ""]
    scope = record.scope or {}
    if scope.get("decision") == "decline":
        out += [f"**Declined.** This request involves {scope['reason']}, which is outside this read-only "
                "public-research system. Please ask a qualified person or rephrase as a general research question.", ""]
        out += [f"_Run: {stats}_"]
        return "\n".join(out)
    if scope.get("decision") == "narrow":
        out += [f"> **Scope note:** this question asks for {scope['reason']}. The brief covers general public "
                "research only and is not personal advice.", ""]
    if record.needs_human_review:
        out += ["> **Needs human review:** " + ("; ".join(record.conflicts) or "evidence was insufficient.")]
        out += [""]
    out += ["## Answer", "", brief.get("answer", ""), ""]
    for title, key in (("Key findings", "key_findings"), ("Limits and conflicting evidence", "limits_and_conflicts"),
                       ("Open questions", "open_questions")):
        items = brief.get(key) or []
        if items:
            out += [f"## {title}", ""] + [f"- {x}" for x in items] + [""]
    if record.gaps:
        out += ["## Gaps disclosed", ""] + [f"- {g}" for g in record.gaps] + [""]

    out += ["## Evidence record", "", "| id | stance | verdict | claim | source passage |", "|---|---|---|---|---|"]
    for f in record.findings:
        c = f.check or {}
        passage = (c.get("passage") or "").replace("|", "/")[:220]
        out.append(f"| {f.id} | {f.stance} | {c.get('verdict', '-')} | {f.claim.replace('|', '/')[:180]} "
                   f"| {passage} |")
    out += ["", "## Sources", ""]
    cited = {f.source_id for f in record.findings}
    for sid in sorted(cited):
        s = record.sources.get(sid, {})
        doi_ok = next((f.check.get("citation", {}) for f in record.findings if f.source_id == sid and f.check), {})
        mark = "DOI verified" if doi_ok.get("title_match") else ("DOI resolves" if doi_ok.get("resolves") else "DOI not verified")
        out.append(f"- **{sid}** {_cite(s)} {s.get('doi') or ''} ({mark})")
    if brief and brief.get("dropped"):
        out += ["", f"_Citation gate removed {len(brief['dropped'])} sentence(s) that cited nothing or cited "
                "unverified evidence._"]
    out += ["", "## How this answer was built", ""]
    for d in record.decisions:
        if d["decision"] == "plan" and "scores" in d:
            out.append(f"- Plan: compared {len(d['candidates'])} decompositions, scores {d['scores']}, chose "
                       f"#{d['chosen'] + 1}. {d['why']}")
        elif d["decision"] == "follow-up":
            out.append(f"- Follow-up: {len(d['candidates'])} next actions scored {d['scores']}, kept {len(d['kept'])}.")
    recov = [t for t in record.trace if t["event"] in ("recover", "guardrail")]
    out.append(f"- {len(record.queries_tried)} searches, {len(record.sources)} sources opened, "
               f"{len(record.findings)} findings checked, {len(recov)} recoveries/guardrail events.")
    out += [f"- Run: {stats}", ""]
    return "\n".join(out)
