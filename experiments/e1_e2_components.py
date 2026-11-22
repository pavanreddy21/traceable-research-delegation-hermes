"""Component experiments on the hand-labeled claim benchmark (eval/claims_benchmark.jsonl, 32 claims).

E1  Claim-checker model selection: which LLM best separates supported from unsupported claims?
E2  Retriever selection: which retriever finds the right evidence passage among ~70 distractors?

    python experiments/e1_e2_components.py            # both
    python experiments/e1_e2_components.py e2         # one
Results go to results/e1_checker_models.json and results/e2_retrievers.json.
"""
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traceable_research import config  # noqa: E402
from traceable_research.agents import check_claim  # noqa: E402
from traceable_research.llm import USAGE  # noqa: E402
from traceable_research.rag import make_retriever, sentences  # noqa: E402
from traceable_research.tools import fetch_source  # noqa: E402

ROOT = config.ROOT
CHECKER_MODELS = [
    "google/gemma-4-12B-it",
    # Qwen/Qwen3.6-35B-A3B was tried but this gateway tier serves it only as background batch jobs
    "zai-org/GLM-5.3-Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
]
RETRIEVERS = ["tfidf", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5",
              "hybrid:BAAI/bge-small-en-v1.5", "hybrid:BAAI/bge-base-en-v1.5"]


def load():
    claims = [json.loads(line) for line in open(ROOT / "eval" / "claims_benchmark.jsonl")]
    sources = {c["source_id"]: fetch_source(c["source_id"]) for c in claims}
    return claims, sources


def e1(claims, sources):
    rows = {}
    for model in CHECKER_MODELS:
        USAGE.reset()
        def one(c):
            t = time.time()
            try:
                chk = check_claim(c["claim"], sources[c["source_id"]], model=model,
                                  retriever=config.EMBED_MODEL)
            except Exception as err:  # model unavailable or quota: record and continue
                chk = {"verdict": "error", "reason": str(err)[:200]}
            return time.time() - t, {"id": c["id"], "label": c["label"], "verdict": chk["verdict"],
                                     "reason": chk.get("reason")}
        with ThreadPoolExecutor(max_workers=4) as pool:
            done = list(pool.map(one, claims))
        lat, out = [d[0] for d in done], [d[1] for d in done]
        pos = [o for o in out if o["label"] == "supported"]
        neg = [o for o in out if o["label"] == "not_supported"]
        accept = lambda o: o["verdict"] in ("supported", "partial")
        tp, fp = sum(map(accept, pos)), sum(map(accept, neg))
        correct = tp + (len(neg) - fp)
        u = USAGE.as_dict()
        rows[model] = {
            "accuracy": round(correct / len(out), 3),
            "true_accept_rate": round(tp / len(pos), 3),          # recall on true claims
            "false_accept_rate": round(fp / len(neg), 3),         # unsupported claims let through (the costly error)
            "errors": sum(o["verdict"] == "error" for o in out),
            "mean_request_seconds": round(u["llm_seconds"] / max(1, u["llm_calls"]), 1),
            "recorded_llm_seconds": u["llm_seconds"], "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"], "items": out}
        print(f"E1 {model}: acc={rows[model]['accuracy']} TAR={rows[model]['true_accept_rate']} "
              f"FAR={rows[model]['false_accept_rate']} errors={rows[model]['errors']}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "e1_checker_models.json").write_text(json.dumps(rows, indent=1))


def e2(claims, sources):
    passages, owner = [], []
    for sid, s in sources.items():
        for p in sentences(s["abstract"]):
            passages.append(p)
            owner.append(sid)
    rows = {}
    for kind in RETRIEVERS:
        ret = make_retriever(passages, kind)
        hit1 = hit3 = rr = 0.0
        for c in claims:
            ranked = [i for i, _ in ret.search(c["claim"], k=10)]
            gold = [i for i, p in enumerate(passages) if c["gold_phrase"].lower() in p.lower()]
            rank = next((r for r, i in enumerate(ranked) if i in gold), None)
            hit1 += rank == 0
            hit3 += rank is not None and rank < 3
            rr += 1 / (rank + 1) if rank is not None else 0
        n = len(claims)
        rows[kind] = {"hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3), "mrr": round(rr / n, 3),
                      "passages": len(passages)}
        print(f"E2 {kind}: {rows[kind]}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "e2_retrievers.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2"]
    claims, sources = load()
    if "e2" in which:
        e2(claims, sources)
    if "e1" in which:
        e1(claims, sources)
