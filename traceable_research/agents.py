"""Role agents. Source finder and counterexample searcher are ReAct tool-use loops; the claim checker is a
fixed verification pipeline (retrieve passage -> hard quote check -> LLM judge -> DOI check)."""
import json
import re

from . import config
from .guardrails import UNTRUSTED_NOTE, BudgetExceeded
from .llm import USAGE, chat, parse_json
from .rag import make_retriever, sentences
from .tools import TOOLS, verify_doi

PROTOCOL = """Tools you can call:
{tools}

Reply with ONE JSON object and nothing else:
{{"thought": "<one short sentence: what you do next and why>", "action": "<tool name or finish>", "args": {{...}}}}

To finish: {{"thought": "...", "action": "finish", "args": {{"findings": [
  {{"claim": "<one specific claim the source makes>", "source_id": "W...", "quote": "<sentence copied exactly from the fetched abstract>", "stance": "{stances}"}}
]}}}}

Rules:
- Search first, then fetch_source for every source you cite. Never cite a source you did not fetch.
- Claims must be specific (population, task, outcome, direction). Copy quotes word for word.
- Give 1 to 3 findings. If nothing credible turns up, finish with an empty list.
- {untrusted}"""

ROLES = {
    "source_finder": {
        "goal": "You are the SOURCE FINDER on a research team. Find peer-reviewed public sources that directly "
                "answer the subquestion and report what they found.",
        "stances": "supports",
    },
    "counterexample": {
        "goal": "You are the COUNTEREXAMPLE SEARCHER on a research team. Look only for credible evidence that "
                "DISAGREES with the common answer, finds no effect, or limits when it holds (other populations, "
                "tasks, delays, settings). Search with contrary phrasings like 'no benefit', 'fails to', 'boundary'.",
        "stances": "challenges|limits",
    },
}


class RoleAgent:
    def __init__(self, role, record, model=None):
        self.role, self.record, self.model = role, record, model or config.AGENT_MODEL
        spec = ROLES[role]
        tools = "\n".join(f"- {n}({', '.join(t['args'])}): {t['desc']}" for n, t in TOOLS.items())
        self.system = spec["goal"] + "\n\n" + PROTOCOL.format(tools=tools, stances=spec["stances"],
                                                              untrusted=UNTRUSTED_NOTE)

    def _obs(self, messages, reply, text):
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Observation: " + text}]

    def run(self, task, hints=""):
        """Return findings as dicts. Every recovery from a misstep is logged to the evidence trace."""
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Research task: {task}{hints}"}]
        fetched = {}
        self.queries = []  # merged into the evidence record by the coordinator in task order (deterministic)
        for step in range(config.MAX_AGENT_STEPS):
            if USAGE.calls >= config.MAX_LLM_CALLS:
                raise BudgetExceeded("task LLM budget reached")
            last = step == config.MAX_AGENT_STEPS - 1
            if last:
                messages.append({"role": "user", "content": "Step limit reached: finish now with what you have."})
            reply = chat(messages, model=self.model, max_tokens=700)
            try:
                move = parse_json(reply)
                action, args = move.get("action"), move.get("args") or {}
            except (ValueError, AttributeError) as err:
                self.record.log(self.role, "recover", problem="unparseable reply", detail=str(err))
                self._obs(messages, reply, f"Your reply was not one valid JSON object ({err}). Reply again.")
                continue
            self.record.log(self.role, "act", step=step, action=action, thought=str(move.get("thought", ""))[:200],
                            args={k: str(v)[:120] for k, v in args.items() if k != "findings"})
            if action == "finish":
                findings = self._accept(args.get("findings") or [], fetched, task)
                if not findings and not fetched and not last:
                    self.record.log(self.role, "recover", problem="finished without fetching a source")
                    self._obs(messages, reply, "You have not fetched any source yet. Search, then fetch_source "
                                               "before you cite anything.")
                    continue
                return findings
            if action not in TOOLS:
                self.record.log(self.role, "recover", problem=f"unknown tool {action!r}")
                self._obs(messages, reply, f"'{action}' is not an available tool. Use one of: "
                                           f"{', '.join(TOOLS)}, or finish.")
                continue
            try:
                result = TOOLS[action]["fn"](**{k: v for k, v in args.items() if k in TOOLS[action]["args"]})
            except TypeError as err:
                result = {"error": f"bad arguments: {err}"}
            if action == "search_papers":
                self.queries.append(str(args.get("query", "")))
                if not result.get("results"):
                    self.record.log(self.role, "recover", problem="search returned nothing")
                    result = {"results": [], "hint": "No results. Try a broader query with different words."}
            if action == "fetch_source" and "error" not in result:
                fetched[result["source_id"]] = result
                self.record.add_source(result)
                if result.get("injection_flags"):
                    self.record.log(self.role, "guardrail", problem="instruction-like text removed from source",
                                    source_id=result["source_id"])
                result = {k: result[k] for k in ("source_id", "title", "year", "authors", "abstract")}
            if "error" in result:
                self.record.log(self.role, "recover", problem=result["error"])
            self._obs(messages, reply, json.dumps(result)[:3500])
        return []

    def _accept(self, findings, fetched, task):
        out = []
        for f in findings[:3]:
            if not isinstance(f, dict):
                continue
            sid = str(f.get("source_id", "")).rsplit("/", 1)[-1]
            if sid not in fetched:
                self.record.log(self.role, "guardrail", problem="cited a source it never fetched", source_id=sid)
                continue
            stance = f.get("stance", "supports")
            if stance not in ("supports", "challenges", "limits"):
                stance = "supports" if self.role == "source_finder" else "limits"
            out.append({"role": self.role, "subquestion": task, "claim": str(f.get("claim", ""))[:400],
                        "source_id": sid, "stance": stance, "quote": str(f.get("quote", ""))[:600]})
        return out


JUDGE = """You check whether a source supports a claim. Use ONLY the source passages below.

Claim: {claim}

Source: {title} ({year})
Passages:
{passages}

Verdicts:
- "supported": a passage states the claim, with the same direction, population/task and outcome.
- "partial": a passage supports a narrower or more qualified version of the claim.
- "not_supported": the passages do not state it, contradict it, or the claim overstates them.

Reply with JSON only: {{"verdict": "supported|partial|not_supported", "passage": <number>, "reason": "<one sentence>"}}"""


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def quote_in_source(quote, abstract):
    """Hard, model-free check that the agent's quote really appears in the fetched text."""
    q, a = _norm(quote).split(), _norm(abstract)
    if len(q) < 5:
        return False
    if " ".join(q[:12]) in a:
        return True
    words = set(a.split())
    return sum(w in words for w in q) / len(q) >= 0.9


def check_claim(claim, source, model=None, retriever=None):
    """Verify one claim against its source. Returns the check dict stored on the finding."""
    passages = sentences(source.get("abstract", ""))
    if not passages:
        return {"verdict": "unverifiable", "reason": "source has no readable text"}
    top = make_retriever(passages, retriever).search(claim, k=3)
    shown = [passages[i] for i, _ in top]
    reply = chat([{"role": "user", "content": JUDGE.format(
        claim=claim, title=source.get("title"), year=source.get("year"),
        passages="\n".join(f"[{n + 1}] {p}" for n, p in enumerate(shown)))}],
        model=model or config.CHECKER_MODEL, max_tokens=200)
    try:
        j = parse_json(reply)
        verdict = j.get("verdict") if j.get("verdict") in ("supported", "partial", "not_supported") else "not_supported"
        idx = int(j.get("passage") or 1) - 1
        passage = shown[idx] if 0 <= idx < len(shown) else shown[0]
        reason = str(j.get("reason", ""))[:300]
    except (ValueError, TypeError, AttributeError):
        verdict, passage, reason = "not_supported", shown[0], "checker reply unparseable; treated as unsupported"
    return {"verdict": verdict, "passage": passage, "reason": reason, "retrieval_score": round(top[0][1], 3)}


def verify_finding(finding, record, model=None, retriever=None):
    source = record.sources.get(finding.source_id, {})
    check = check_claim(finding.claim, source, model, retriever)
    check["quote_found"] = quote_in_source(finding.quote, source.get("abstract", ""))
    check["citation"] = verify_doi(source.get("doi"), source.get("title", ""))
    if check["verdict"] == "supported" and not check["citation"]["resolves"]:
        check["verdict"] = "partial"
        check["reason"] += " (citation DOI did not resolve)"
    finding.check = check
    record.log("claim_checker", "check", finding=finding.id, verdict=check["verdict"],
               quote_found=check["quote_found"], doi_ok=check["citation"]["resolves"])
    return finding
