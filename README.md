# Traceable Research Delegation for Hermes

A capstone project proposing an evidence-review workflow for public-information research with Hermes Agent. The intended user is a student or reader who wants a concise synthesis and a clear path from each important claim back to its source.

## Project status

The repository contains one runnable component: a small local semantic-retrieval demonstration over five source-linked, paraphrased notes from two public studies. It uses TF–IDF, a three-dimensional latent semantic indexing projection with NumPy SVD, a small query-alias map, and cosine similarity to rank notes.

The coordinator, source finder, claim checker, counterexample searcher, live web retrieval, shared evidence state, Tree of Thought controller, and safety gates are proposed extensions. They are not implemented by the local script. The script uses fixed notes and does not fetch pages.

## Repository contents

- `RAG_Semantic_Retrieval_Demo.py` — local retrieval demonstration.
- `requirements.txt` — NumPy dependency.
- `docs/Final_Capstone_Report_Hermes.pdf` — final report with architecture, design evolution, evaluation criteria, results, limitations, and safety plan.

The presentation deck references this repository. The recorded presentation is submitted separately in Canvas.

## Run the demonstration

Use Python 3.9 or later. From the repository directory, create and activate a virtual environment, then install the dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python RAG_Semantic_Retrieval_Demo.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The script prints the top-ranked notes for two queries. It is deterministic for a fixed Python and NumPy environment. The reported cosine values rank notes within this five-item corpus; they are not probabilities of truth, confidence ratings, or benchmark scores.

## Worked example

The demonstration asks whether self-testing instead of rereading improves memory for facts on a later exam. One query ranks a prose delayed-recall note first (reported cosine 0.806), followed by a math study result (0.614) and its methods note (0.591). A focused counterevidence query ranks the math methods note at 0.935 and the no-reliable-benefit result at 0.854. The studies cover different tasks, so the synthesis is task-specific rather than a universal claim about exam performance.

## Evaluation and limitations

The worked run illustrates retrieval and evidence contrast. It is not a formal benchmark. The corpus has five hand-written notes from two papers, and the example covers one research question. The project has not measured material-claim support rates, citation-link success across a test set, confidence calibration, latency, or full-system reliability. The proposed pilot plan defines those measures and target gates in the report.

## Sources

- Roediger, H. L., & Karpicke, J. D. (2006). *Test-enhanced learning: Taking memory tests improves long-term retention.* Psychological Science, 17(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x
- Huang, X., Zheng, S., Yu, Z., & Chen, S. (2023). *Retrieval practice may not benefit mathematical word-problem solving.* Frontiers in Psychology, 14, 1093653. https://doi.org/10.3389/fpsyg.2023.1093653

## Public project link

https://github.com/pavanreddy21/traceable-research-delegation-hermes
