"""Retrieval-augmented few-shot: find verified question->SQL examples similar to the new question.

Embedders
- tfidf     : pure-Python TF-IDF, no downloads (offline tests, CI fallback)
- fastembed : BAAI/bge-small-en-v1.5 dense embeddings via fastembed (pip install fastembed)
The index is a small in-memory cosine store; in production swap for pgvector / Azure AI Search.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
BANK = ROOT / "data" / "example_bank.jsonl"
STOP = set("a an the of in on for to and or by is are was were what which how many much each per with from as at "
           "do does did be give return show list".split())


def _tokens(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]          # unigrams + bigrams


# --------------------------------------------------------------------------- embedders
class TfidfEmbedder:
    name = "tfidf"

    def fit(self, docs: list[str]):
        df = Counter(t for d in docs for t in set(_tokens(d)))
        self.idf = {t: math.log((1 + len(docs)) / (1 + c)) + 1 for t, c in df.items()}
        return self

    def embed(self, texts: list[str]) -> list[dict]:
        out = []
        for t in texts:
            tf = Counter(_tokens(t))
            v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0) for w, c in tf.items()}
            n = math.sqrt(sum(x * x for x in v.values())) or 1.0
            out.append({w: x / n for w, x in v.items()})
        return out

    @staticmethod
    def cosine(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(w, 0.0) for w, x in a.items())


class FastEmbedEmbedder:
    name = "fastembed:bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def fit(self, docs):
        return self

    def embed(self, texts):
        return [list(map(float, v)) for v in self.model.embed(texts)]          # already L2-normalised

    @staticmethod
    def cosine(a, b) -> float:
        return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- index + retriever
class Retriever:
    def __init__(self, kind: str = "tfidf", bank_path: Path = BANK):
        self.examples = [json.loads(l) for l in bank_path.read_text().splitlines() if l.strip()]
        docs = [e["question"] for e in self.examples]
        self.emb = (FastEmbedEmbedder() if kind == "fastembed" else TfidfEmbedder()).fit(docs)
        self.vectors = self.emb.embed(docs)                                     # the "vector store"
        self.name = self.emb.name

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        q = self.emb.embed([query])[0]
        scored = sorted(((e, self.emb.cosine(q, v)) for e, v in zip(self.examples, self.vectors)),
                        key=lambda x: -x[1])
        return scored[:k]

    def leakage(self, cases: list[dict], threshold: float = 0.85) -> list[tuple[str, str, float]]:
        """Test questions that are near-duplicates of bank examples (would inflate RAG results)."""
        hits = []
        for c in cases:
            e, s = self.search(c["question"], 1)[0]
            if s >= threshold:
                hits.append((c["id"], e["id"], round(s, 3)))
        return hits


def format_examples(hits) -> str:
    lines = ["", "Verified examples of similar questions (follow the pattern, not the exact values):"]
    for e, _ in hits:
        lines.append(f"Q: {e['question']}")
        lines.append(f"SQL: {e['sql']}" if e.get("sql") else f"Expected behaviour: {e['answer']}")
    return "\n".join(lines)


class RAGAgent:
    """Wraps any agent: per question, appends the top-k retrieved examples to its system prompt."""

    def __init__(self, inner, retriever: Retriever, k: int = 3):
        self.inner, self.retriever, self.k = inner, retriever, k
        self.base_prompt = getattr(inner, "system_prompt", None) or getattr(inner, "prompt", "")
        self.name = f"{getattr(inner, 'name', 'agent')}+rag({retriever.name},k={k})"

    def run(self, question: str):
        hits = self.retriever.search(question, self.k)
        prompt = self.base_prompt + "\n" + format_examples(hits)
        if hasattr(self.inner, "system_prompt"):
            self.inner.system_prompt = prompt
        if hasattr(self.inner, "prompt"):                                     # MockAgent
            self.inner.prompt = prompt.lower()
        out = self.inner.run(question)
        out.retrieved = [e["id"] for e, _ in hits]
        return out
