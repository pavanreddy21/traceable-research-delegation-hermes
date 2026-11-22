"""Passage retrieval (the RAG layer). Two interchangeable retrievers, compared in experiment E2:

- TfidfRetriever: the checkpoint 3.1 prototype method (TF-IDF + cosine), no model needed.
- EmbeddingRetriever: local sentence embeddings (fastembed: BGE / MiniLM) + cosine.
"""
import math
import re
from collections import Counter

import numpy as np

from . import config
from .llm import embed

STOP = set("a an and are as at be but by for from had has have in into is it its of on or over same the their "
           "then this to was were with after when how does did than that these those which who we our".split())


def sentences(text, max_words=60):
    """Split an abstract into passages of one or two sentences."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]
    out, buf = [], ""
    for s in parts:
        buf = f"{buf} {s}".strip()
        if len(buf.split()) >= 25 or len(buf.split()) > max_words:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, passages):
        self.passages = passages
        counts = [Counter(_tokens(p)) for p in passages]
        vocab = sorted({t for c in counts for t in c})
        self.index = {t: i for i, t in enumerate(vocab)}
        n = len(passages)
        self.idf = np.array([math.log((1 + n) / (1 + sum(t in c for c in counts))) + 1 for t in vocab])
        self.matrix = np.array([self._vec(c) for c in counts]) if passages else np.zeros((0, 0))

    def _vec(self, counts):
        v = np.zeros(len(self.index))
        for t, c in counts.items():
            if t in self.index:
                v[self.index[t]] = (1 + math.log(c)) * self.idf[self.index[t]]
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def search(self, query, k=3):
        if not self.passages:
            return []
        scores = self.matrix @ self._vec(Counter(_tokens(query)))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


class EmbeddingRetriever:
    def __init__(self, passages, model=None):
        self.model = model or config.EMBED_MODEL
        self.name = self.model.rsplit("/", 1)[-1]
        self.passages = passages
        self.matrix = self._norm(np.array(embed(passages, self.model))) if passages else np.zeros((0, 0))

    @staticmethod
    def _norm(m):
        return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)

    def search(self, query, k=3):
        if not self.passages:
            return []
        q = self._norm(np.array(embed([query], self.model)[0]))
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


class HybridRetriever:
    """Reciprocal-rank fusion of TF-IDF (exact terms, numbers) and embeddings (paraphrase)."""

    def __init__(self, passages, model=None):
        self.parts = [TfidfRetriever(passages), EmbeddingRetriever(passages, model)]
        self.name = "hybrid:" + self.parts[1].name
        self.passages = passages

    def search(self, query, k=3):
        fused = {}
        for part in self.parts:
            for rank, (i, _) in enumerate(part.search(query, k=len(self.passages))):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(i, round(fused[i] * 30, 3)) for i in order]


def make_retriever(passages, kind=None):
    kind = kind or config.EMBED_MODEL
    if kind == "tfidf":
        return TfidfRetriever(passages)
    if kind.startswith("hybrid:"):
        return HybridRetriever(passages, kind.split(":", 1)[1])
    return EmbeddingRetriever(passages, kind)
