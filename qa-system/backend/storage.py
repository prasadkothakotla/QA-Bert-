"""
In-memory document store.

Swap this for a real database (Postgres, SQLite, etc.) when you move past
prototyping -- the interface (add_document / get_document / list_documents /
delete_document) is the part worth keeping stable.
"""
import re
import uuid
import time
from typing import Optional


class Chunk:
    def __init__(self, chunk_id: str, doc_id: str, text: str, start_offset: int):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.start_offset = start_offset  # char offset into the original document


class Document:
    def __init__(self, doc_id: str, title: str, text: str):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.uploaded_at = time.time()
        self.chunks: list[Chunk] = []


class DocumentStore:
    def __init__(self):
        self._docs: dict[str, Document] = {}

    def add_document(self, title: str, text: str, chunk_size: int = 800, overlap: int = 150) -> Document:
        doc_id = str(uuid.uuid4())
        doc = Document(doc_id, title, text)
        doc.chunks = self._chunk_text(doc_id, text, chunk_size, overlap)
        self._docs[doc_id] = doc
        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._docs.get(doc_id)

    def list_documents(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.uploaded_at, reverse=True)

    def delete_document(self, doc_id: str) -> bool:
        return self._docs.pop(doc_id, None) is not None

    def all_chunks(self) -> list[Chunk]:
        chunks = []
        for doc in self._docs.values():
            chunks.extend(doc.chunks)
        return chunks

    @staticmethod
    def _chunk_text(doc_id: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Sliding-window chunking on whitespace-split words, tracking char offsets
        so the frontend can highlight the exact answer span later."""
        # `split(" ")` treats a whole newline-delimited document as one word,
        # which makes retrieval effectively useless for most PDFs and notes.
        # Keep words and their original positions so chunks remain compact while
        # allowing arbitrary whitespace in uploaded source material.
        word_matches = list(re.finditer(r"\S+", text))
        chunks = []
        i = 0
        if not word_matches:
            return chunks

        step = max(1, chunk_size - overlap)
        for start in range(0, len(word_matches), step):
            end = min(start + chunk_size, len(word_matches))
            chunk_words = [match.group(0) for match in word_matches[start:end]]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            start_offset = word_matches[start].start()
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{start}",
                doc_id=doc_id,
                text=chunk_text,
                start_offset=start_offset,
            ))
            if end == len(word_matches):
                break
        return chunks


store = DocumentStore()
