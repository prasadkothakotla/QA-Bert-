"""
Sparse (TF-IDF) retrieval over document chunks.

This is the "search across uploaded docs" half of the pipeline: given a
question, find the N most relevant chunks before handing them to BERT for
extraction. TF-IDF is a reasonable, dependency-light default for a
prototype. If you outgrow it, swap this module for embeddings + a vector
index (FAISS, Chroma, pgvector) -- the rest of the app doesn't need to
change, since it only calls `retrieve(question, top_k)`.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from storage import store, Chunk


class Retriever:
    def __init__(self):
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._chunk_refs: list[Chunk] = []
        self._dirty = True

    def mark_dirty(self):
        """Call after any document is added or removed."""
        self._dirty = True

    def _rebuild_index(self):
        self._chunk_refs = store.all_chunks()
        if not self._chunk_refs:
            self._vectorizer = None
            self._matrix = None
            self._dirty = False
            return
        texts = [c.text for c in self._chunk_refs]
        # Do not remove every word in small/simple documents. If the English
        # stop-word pass leaves no vocabulary, fall back to token-only TF-IDF.
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        try:
            self._matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            self._vectorizer = TfidfVectorizer(max_features=20000)
            self._matrix = self._vectorizer.fit_transform(texts)
        self._dirty = False

    def retrieve(self, question: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if self._dirty:
            self._rebuild_index()
        if self._matrix is None or self._vectorizer is None:
            return []

        question = question.strip()
        if not question:
            return []
        q_vec = self._vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        scored = list(zip(self._chunk_refs, sims))
        if doc_ids:
            scored = [(c, s) for c, s in scored if c.doc_id in doc_ids]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


retriever = Retriever()
