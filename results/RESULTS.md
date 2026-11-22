# Results

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

