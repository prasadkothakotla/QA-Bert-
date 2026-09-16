from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from storage import store
from retriever import retriever
from qa_engine import engine
from pdf_reader import extract_text_from_pdf

app = FastAPI(title="Document QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class DocumentOut(BaseModel):
    doc_id: str
    title: str
    num_chunks: int
    uploaded_at: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    doc_ids: list[str] | None = None  # limit search to these docs; None = whole corpus
    top_k_chunks: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    score: float
    doc_id: str
    doc_title: str
    context: str
    answer_start: int  # offset within `context`
    answer_end: int


# ---------- Routes ----------

@app.post("/documents/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "A filename is required.")
    filename = file.filename.lower()
    if not filename.endswith((".txt", ".md", ".pdf")):
        raise HTTPException(400, "Only .txt, .md, and .pdf files are supported right now.")

    raw = await file.read()

    if filename.endswith(".pdf"):
        try:
            text = extract_text_from_pdf(raw)
        except Exception:
            raise HTTPException(400, "Could not read this PDF. It may be corrupted or encrypted.")
        if not text.strip():
            raise HTTPException(400, "No extractable text found in this PDF. "
                                      "If it's a scanned document, it needs OCR first.")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(400, "Could not decode file as UTF-8 text.")
        if not text.strip():
            raise HTTPException(400, "File is empty.")

    doc = store.add_document(title=file.filename, text=text)
    retriever.mark_dirty()

    return DocumentOut(
        doc_id=doc.doc_id,
        title=doc.title,
        num_chunks=len(doc.chunks),
        uploaded_at=doc.uploaded_at,
    )


@app.get("/documents", response_model=list[DocumentOut])
def list_documents():
    return [
        DocumentOut(doc_id=d.doc_id, title=d.title, num_chunks=len(d.chunks), uploaded_at=d.uploaded_at)
        for d in store.list_documents()
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    ok = store.delete_document(doc_id)
    if not ok:
        raise HTTPException(404, "Document not found.")
    retriever.mark_dirty()
    return {"deleted": doc_id}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not store.list_documents():
        raise HTTPException(400, "Upload at least one document before asking a question.")

    try:
        candidates = retriever.retrieve(req.question, top_k=req.top_k_chunks, doc_ids=req.doc_ids)
    except ValueError as exc:
        raise HTTPException(400, f"Could not index the uploaded text: {exc}") from exc
    if not candidates:
        raise HTTPException(404, "No relevant passages found.")

    best = None
    for chunk, retrieval_score in candidates:
        try:
            result = engine.answer(req.question, chunk.text)
        except RuntimeError as exc:
            raise HTTPException(503, f"Question-answering model is unavailable: {exc}") from exc
        # Combine retrieval relevance with extraction confidence so a
        # high-confidence answer from a marginally-relevant chunk doesn't
        # always beat a lower-confidence answer from the best-matching one.
        combined = 0.4 * retrieval_score + 0.6 * result["score"]
        if best is None or combined > best["combined"]:
            doc = store.get_document(chunk.doc_id)
            best = {
                "combined": combined,
                "answer": result["answer"],
                "score": result["score"],
                "doc_id": chunk.doc_id,
                "doc_title": doc.title if doc else "Unknown",
                "context": chunk.text,
                "answer_start": result["start"],
                "answer_end": result["end"],
            }

    if best is None or not best["answer"].strip():
        raise HTTPException(404, "Could not find an answer in the uploaded documents.")

    return AskResponse(
        answer=best["answer"],
        score=best["score"],
        doc_id=best["doc_id"],
        doc_title=best["doc_title"],
        context=best["context"],
        answer_start=best["answer_start"],
        answer_end=best["answer_end"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(store.list_documents())}
