"""The shared evidence record: the one place every agent writes to and the coordinator reads from."""
import json
import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    id: str
    role: str            # source_finder | counterexample
    subquestion: str
    claim: str
    source_id: str
    stance: str          # supports | challenges | limits
    quote: str = ""      # passage the role agent says backs the claim
    check: dict = field(default_factory=dict)  # filled by the claim checker


@dataclass
class EvidenceRecord:
    question: str
    scope: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)      # source_id -> metadata + abstract
    findings: list = field(default_factory=list)
    queries_tried: list = field(default_factory=list)
    decisions: list = field(default_factory=list)    # visible decision log (Tree of Thought scores), not hidden CoT
    trace: list = field(default_factory=list)        # tool calls and recoveries, per agent
    gaps: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    needs_human_review: bool = False
    started: float = field(default_factory=time.time)

    def __post_init__(self):
        self._lock = threading.Lock()

    def add_source(self, rec):
        with self._lock:
            self.sources.setdefault(rec["source_id"], rec)

    def add_finding(self, **kw):
        with self._lock:
            f = Finding(id=f"E{len(self.findings) + 1}", **kw)
            self.findings.append(f)
            return f

    def log(self, agent, event, **detail):
        with self._lock:
            self.trace.append({"t": round(time.time() - self.started, 2), "agent": agent, "event": event, **detail})

    def verified(self):
        return [f for f in self.findings if f.check.get("verdict") in ("supported", "partial")]

    def to_dict(self):
        d = asdict(self)
        d.pop("_lock", None)
        return d

    def save(self, path):
        path.write_text(json.dumps(self.to_dict(), indent=1, default=str))
