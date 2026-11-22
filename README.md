# Traceable Research Delegation for Hermes

A multi-agent research assistant that answers a question from public scholarly sources and shows where
every claim comes from. Final capstone project (Research Assistant option).

- **User:** a student or reader who wants a short, sourced answer they can verify.
- **Output:** a brief where every factual sentence cites a checked evidence id, plus conflicting evidence,
  disclosed gaps, a source list with verified DOIs, and a "needs human review" flag when warranted.
- **Report:** [`docs/Final_Capstone_Report_Hermes.pdf`](docs/Final_Capstone_Report_Hermes.pdf) ·
  **Slides:** [`docs/Capstone_Slides_Hermes.pdf`](docs/Capstone_Slides_Hermes.pdf) · video submitted in Canvas.

## Quick start

Python 3.9+. No `.venv` is committed; the setup script creates it.

```bash
./setup.sh                               # venv + dependencies + .env template + offline tests
source .venv/bin/activate
TRACE_MODE=replay ./experiments/run_all.sh   # reproduce every saved result: no key, no network
```

Windows PowerShell: `.\setup.ps1`, then `.\.venv\Scripts\Activate.ps1`.

Live run on a new question (needs any OpenAI-compatible endpoint in `.env`, see `.env.example`):

```bash
python run.py "Does spacing study sessions improve long-term retention?"
python run.py --interactive "..."        # lets the planner ask one clarifying question
python run.py --remember "prefer meta-analyses"   # long-term memory is written only on request
```

Each run writes `runs/<slug>/report.md` (the brief) and `evidence.json` (the full evidence record and trace).

## How it works

```
question ─► scope gate ─► coordinator (Tree-of-Thought plan: 3 plans, score, keep best)
                               │
               ┌───────────────┴────────────────┐   run in parallel (ReAct tool loops)
        source finder × N                counterexample searcher
               └───────────────┬────────────────┘
                         evidence record ◄── claim checker: passage retrieval (bge-base),
                               │              quote check, LLM judge, Crossref DOI check
                       coverage review ── gaps? ToT follow-ups (3 proposed, keep 2, ≤ 3 rounds)
                               │
                 writer + citation gate ─► cited brief · conflicts · gaps · human-review flag
```

| Course concept | Where it is |
|---|---|
| Tool calling | `tools.py`: `search_papers`, `fetch_source` (OpenAlex), `verify_doi` (Crossref); JSON action protocol in `agents.py` |
| ReAct + recovery | `agents.py` `RoleAgent.run`: malformed reply, unknown tool, empty search, citing an unopened source → corrective observation |
| RAG / memory | `rag.py` passage retrieval; `evidence.py` task record; `memory.py` past runs (hints, re-verified) + preferences (on approval) |
| Tree of Thought | `coordinator.py` `plan` and `_followups`: propose 3, score with a rubric, keep best / top 2, max 3 rounds |
| Multi-agent | `coordinator.py` `_run_agents`: source finders + counterexample searcher in a thread pool; claim checker after |
| Safety | `guardrails.py` scope gate + untrusted-text filter; citation gate in `coordinator._gate`; budgets in `config.py` |

## Repository layout

| Path | What |
|---|---|
| `traceable_research/` | the system (≈900 lines of Python) |
| `run.py` | command-line entry point |
| `eval/` | 8 research questions, 10 safety cases, 32 hand-labeled claims, `run_eval.py` |
| `experiments/` | `e1_e2_components.py` (model + retriever selection), `run_all.sh` |
| `results/` | saved metrics (`e1_…json` … `e4_…json`) and every evaluated report under `results/runs/` |
| `cache/` | recorded LLM / embedding / HTTP responses that make replay exact |
| `tests/` | offline tests: a scripted fake LLM drives the real coordinator |
| `hermes/traceable-research/SKILL.md` | Hermes Agent skill wrapper (copy to `~/.hermes/skills/`) |
| `RAG_Semantic_Retrieval_Demo.py` | the checkpoint 3.1 retrieval prototype, kept for history |

## Experiments and results

<!-- results:start -->
### E1: claim-checker model (32 hand-labeled claims, `eval/claims_benchmark.jsonl`)

| Model | Accuracy | True claims accepted | False claims accepted | Mean request time | Tokens |
|---|---|---|---|---|---|
| google/gemma-4-12B-it | 97% | 94% | 0% | 7.6 s | 12k |
| zai-org/GLM-5.3-Flash | 97% | 100% | 7% | 6.0 s | 17k |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 100% | 100% | 0% | 4.7 s | 16k |
| deepseek-ai/DeepSeek-V4-Pro-0813 | 97% | 100% | 7% | 2.4 s | 11k |

### E2: retriever (same claims vs. 51 passages)

| Retriever | hit@1 | hit@3 | MRR |
|---|---|---|---|
| tfidf | 69% | 84% | 0.802 |
| sentence-transformers/all-MiniLM-L6-v2 | 62% | 88% | 0.759 |
| BAAI/bge-small-en-v1.5 | 69% | 81% | 0.784 |
| BAAI/bge-base-en-v1.5 | 75% | 94% | 0.834 |
| hybrid:BAAI/bge-small-en-v1.5 | 69% | 91% | 0.810 |
| hybrid:BAAI/bge-base-en-v1.5 | 75% | 91% | 0.845 |

### E3: end to end on 8 questions (`eval/questions.jsonl`), full system vs. no counterexample agent

| Mean over questions | Full | No counterexample |
|---|---|---|
| Findings that passed checking | 93% | 91% |
| Agent quotes found verbatim in source | 100% | 100% |
| Cited DOIs verified (Crossref / DataCite) | 99% | 100% |
| Writer sentences citing verified evidence (before gate) | 99% | 100% |
| Sentences removed by citation gate (total) | 1 | 0 |
| Subquestions with verified support | 96% | 96% |
| Briefs citing counter-evidence | 8/8 | 0/8 |
| Briefs flagged for human review | 6/8 | 0/8 |
| Verified findings / question | 10.1 | 7.5 |
| LLM calls / question | 39.8 | 31.4 |
| LLM request time / question (s, summed) | 375 | 318 |

Every brief and evidence record: `results/runs/<config>/<question>/`.

### E4: scope gate (10 seeded cases, `eval/safety_cases.jsonl`)

| Layer | Accuracy | Missed |
|---|---|---|
| rules only | 80% | S7, S8 |
| rules plus llm | 100% | none |

<!-- results:end -->

### Key trade-offs and why the final approach

- **Checker model → DeepSeek V4.1 Flash (E1).** Only model with 0 false accepts *and* 0 false rejects. DeepSeek V4 Pro
  was faster and cheaper but accepted a claim with swapped numbers; GLM Flash accepted an overstated claim; Gemma 12B
  never accepted a false claim but broke the JSON format once. False accepts are the costly error, so accuracy won over speed.
- **Retriever → BGE-base (E2).** The checker reads the top 3 passages, so hit@3 decides: BGE-base 94% vs TF-IDF 84%
  (the checkpoint 3.1 prototype). A TF-IDF + BGE hybrid had the best MRR but a lower hit@3, so the simpler model won.
- **Counterexample agent stays (E3).** +27% LLM calls (39.8 vs 31.4), but it is the only source of counter-evidence
  (8/8 briefs vs 0/8) and adds 2.6 verified findings per question. Without it the briefs look cleaner and are never flagged, because the
  disagreement is simply missing.
- **Tree of Thought stays, bounded.** Plans are scored and the best kept on every question; follow-up rounds fired when
  coverage had gaps (Q2 used all 3 rounds). Kept to 3 branches / 2 kept / 3 rounds for cost.
- **Scope gate → rules + LLM layer (E4).** Rules alone missed two reworded requests (80%); the LLM layer fixed both
  (100%) and can only make a decision stricter.
- **DOI check fixed after the first run.** It reported 91%: three false failures were checker bugs (HTML entities,
  Crossref storing only the main title) and two were DataCite DOIs. After the fix: 99%. The one remaining failure is a
  source with no DOI, which stays counted as a failure.
- **Plain Python over CrewAI / LangChain / MCP.** Same parallelism with a thread pool, fewer dependencies, a full trace
  per run, and exact record-and-replay. Trade-off: no built-in persistence or dashboards.

### Manual spot-check and memory demo

- **Spot-check** (`results/spot_check.csv`): 20 verified findings sampled evenly across questions and read against their
  passages. 20/20 have the core claim supported; in 1 the checker said "supported" where "partial" is fairer
  (a condition not in the passage); 2 carried the wrong stance label (limiting evidence tagged "supports"), though the
  briefs still presented them as limits; 1 was off-topic for its question.
- **Memory** (`experiments/memory_demo.sh`): the second, related question recalled the first run's verified sources as
  hints (`results/memory_demo/run2/evidence.json`, event `memory`), and re-fetched and re-checked them.
- **Exact replay verified:** `TRACE_MODE=replay` with no key and no network reproduces every results file. This check
  caught a bug (parallel agents logged tried queries in random order, so follow-up prompts could not be replayed); it
  was fixed and Q2, the only question with follow-ups, was re-recorded.
- **Recovery in action (Q2):** 18 empty searches were retried with new wording, 2 invented source ids failed to fetch
  and were recovered from, and 3 attempts to cite unopened sources were rejected; the one subquestion still without
  verified support is disclosed as a gap in the brief.
- **Human-review flag** fired on 6/8 mixed-evidence questions, always because the writer reported a contradiction it could
  not explain. It is conservative by design; calibrating it is future work.

## Safety: unintended actions and controls

| Possible unintended action | Control |
|---|---|
| Acting in the world (send, post, buy, delete) | Agents only have read-only tools; such requests are declined by the scope gate |
| Revealing private information about a person | Declined by rules before any model call |
| Giving individual medical, legal or financial advice | Narrowed to general research; the brief carries a scope note |
| Following instructions hidden in a source page | Instruction-like sentences are removed and flagged; the run is marked for human review |
| Citing invented or mismatched sources | Agents must fetch before citing; quote check; Crossref DOI check; citation gate drops unverified sentences |
| Runaway cost or loops | Caps: 6 steps per agent, 3 review rounds, 80 LLM calls per task, rate limiter |
| Treating old findings as current truth | Memory gives hints only; every source is re-fetched and re-checked |

## Limitations

- Small evaluation: 8 questions and 32 claims, labeled by the author; the numbers show behavior, not population-level reliability.
- Abstracts only: the checker reads OpenAlex abstracts, not full texts, so claims about details in the body cannot be verified.
- The claim checker is an LLM judge; E1 measures how often it is wrong.
- The rule-based scope gate misses rephrased requests (E4); the LLM layer helps but is not a guarantee.

## Sources used in examples

Data comes from [OpenAlex](https://openalex.org) (CC0) and [Crossref](https://www.crossref.org). Papers are cited in each report.
