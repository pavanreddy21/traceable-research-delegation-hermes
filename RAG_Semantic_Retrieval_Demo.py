"""Small, reproducible semantic-retrieval prototype for the capstone example.

Requires NumPy. The five source-linked records below are concise, paraphrased
chunks captured from the cited public research pages after source verification.
This demo shows local indexing and retrieval; it does not fetch the papers live.
"""
import math
import re
from collections import Counter
import numpy as np

CHUNKS = [
    {"id": "RK06-A", "source": "Roediger & Karpicke (2006)", "url": "https://doi.org/10.1111/j.1467-9280.2006.01693.x", "location": "Abstract", "text": "Students studied prose passages, practiced free recall without feedback, or restudied for the same number of periods."},
    {"id": "RK06-B", "source": "Roediger & Karpicke (2006)", "url": "https://doi.org/10.1111/j.1467-9280.2006.01693.x", "location": "Abstract", "text": "A five minute final test favored rereading, but testing led to better recall after two days and one week."},
    {"id": "HC23-A", "source": "Huang et al. (2023)", "url": "https://doi.org/10.3389/fpsyg.2023.1093653", "location": "Abstract", "text": "The study tested worked examples and mathematical word problems, comparing restudy with retrieval practice through example problem pairs."},
    {"id": "HC23-B", "source": "Huang et al. (2023)", "url": "https://doi.org/10.3389/fpsyg.2023.1093653", "location": "Abstract and General Discussion", "text": "Across three experiments, retrieval practice did not reliably improve delayed mathematical problem solving over restudy."},
    {"id": "HC23-C", "source": "Huang et al. (2023)", "url": "https://doi.org/10.3389/fpsyg.2023.1093653", "location": "Methods: Participants and Materials", "text": "Researchers varied practice difficulty, example materials, and feedback; university students solved Grade 6 math problems."},
]
STOP = set("a an and are as at be but by for from had has have in into is it of on or over same the their then this to was were with after when how does did than that".split())
ALIASES = {
    "self testing": "retrieval practice recall",
    "practice testing": "retrieval practice",
    "rereading": "restudy",
    "re-reading": "restudy",
    "reading again": "restudy",
    "memory": "retention",
    "later exam": "delayed final test",
    "after a delay": "delayed test",
    "word problems": "mathematical problem solving",
}

def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]

def expand_query(query):
    expanded = query.lower()
    for phrase, addition in ALIASES.items():
        if phrase in expanded:
            expanded += " " + addition
    return expanded

TERMS = sorted({term for chunk in CHUNKS for term in tokens(chunk["text"])})
TERM_INDEX = {term: i for i, term in enumerate(TERMS)}
COUNTS = [Counter(tokens(chunk["text"])) for chunk in CHUNKS]
IDF = np.array([math.log((1 + len(CHUNKS)) / (1 + sum(term in row for row in COUNTS))) + 1 for term in TERMS])
TFIDF = np.zeros((len(CHUNKS), len(TERMS)))
for row_index, row in enumerate(COUNTS):
    for term, count in row.items():
        TFIDF[row_index, TERM_INDEX[term]] = (1 + math.log(count)) * IDF[TERM_INDEX[term]]

# Latent semantic indexing: project TF-IDF rows into three SVD dimensions.
U, S, VT = np.linalg.svd(TFIDF, full_matrices=False)
DIMENSIONS = min(3, len(S))
DOCUMENT_VECTORS = U[:, :DIMENSIONS] * S[:DIMENSIONS]

def retrieve(query, top_k=5):
    counts = Counter(tokens(expand_query(query)))
    vector = np.zeros(len(TERMS))
    for term, count in counts.items():
        if term in TERM_INDEX:
            vector[TERM_INDEX[term]] = (1 + math.log(count)) * IDF[TERM_INDEX[term]]
    query_vector = vector @ VT[:DIMENSIONS].T
    denominator = np.linalg.norm(DOCUMENT_VECTORS, axis=1) * np.linalg.norm(query_vector) + 1e-12
    scores = (DOCUMENT_VECTORS @ query_vector) / denominator
    return sorted(zip(scores, CHUNKS), key=lambda item: item[0], reverse=True)[:top_k]

def show(query):
    print(f"\nQUERY: {query}")
    for rank, (score, chunk) in enumerate(retrieve(query), 1):
        print(f"{rank}. {chunk['id']} | cosine={score:.3f} | {chunk['source']} | {chunk['location']} | {chunk['text']}")

if __name__ == "__main__":
    print("Index: 5 source-linked chunks | TF-IDF projected to 3 LSI dimensions | cosine ranking")
    print("Query expansion: self-testing to retrieval practice; rereading to restudy; later exam to delayed test")
    show("Does self testing instead of rereading improve memory for facts on a later exam?")
    show("Does practice testing help students solve math word problems after a delay?")
