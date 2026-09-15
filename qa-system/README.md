# Document QA System

A corpus search-and-answer tool: upload text documents, ask a question, and get
back an extracted answer with a confidence score and the exact source passage
highlighted — built on a retrieve-then-read pipeline (TF-IDF retrieval + BERT
extractive QA).

```
qa-system/
├── backend/
│   ├── main.py         FastAPI app — upload, list, delete, ask
│   ├── qa_engine.py     <- PLUG YOUR EXISTING BERT MODEL IN HERE
│   ├── retriever.py     TF-IDF search across document chunks
│   ├── storage.py       in-memory document store + chunking
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api.js
    │   └── components/
    │       ├── DocumentLibrary.jsx   upload + index-card list, left rail
    │       ├── QuestionBar.jsx       question input
    │       └── AnswerPanel.jsx       answer + confidence + highlighted source
    └── package.json
```

## How the pipeline works

1. **Upload** — a document is split into overlapping ~800-word chunks (`storage.py`).
2. **Retrieve** — on a question, TF-IDF cosine similarity ranks the most relevant
   chunks across the corpus (or just the documents you've selected in the UI)
   (`retriever.py`).
3. **Read** — your BERT model extracts the answer span from each top candidate
   chunk (`qa_engine.py`). The best-scoring result (blending retrieval relevance
   and extraction confidence) is returned.
4. **Display** — the frontend highlights the exact extracted span inside the
   source passage, like a highlighter mark, so you can see it in context.

## Your model

Your fine-tuned `BertForQuestionAnswering` model (from Colab's `save_pretrained()`)
is already wired in and sitting at `backend/bert_qa_model/`:

```
backend/bert_qa_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

`qa_engine.py` loads it straight from that folder with `local_files_only=True`,
so it never touches the Hugging Face Hub. If you retrain and export a new
version, just drop the new files into that same folder — no code changes
needed.

`model.safetensors` is ~415MB. That's fine to run locally, but if you push this
to git, use [Git LFS](https://git-lfs.com/) for that file rather than committing
it as a normal blob.

## Running it

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). The frontend expects
the API at `http://localhost:8000` — change `BASE_URL` in `frontend/src/api.js`
if yours runs elsewhere.

## Notes / next steps

- Upload accepts `.txt`, `.md`, and `.pdf` (text extracted via `pypdf` in
  `pdf_reader.py`). Scanned/image-only PDFs have no text layer to extract —
  those need OCR (e.g. `pytesseract`) first. Add a `.docx` extractor the same
  way if you need it.
- Document storage is in-memory and resets on backend restart — swap `storage.py`
  for a real database when you're past prototyping.
- TF-IDF retrieval is a solid, dependency-light default. If your corpus grows
  large or you want semantic (not just keyword) matching, swap `retriever.py`
  for embeddings + a vector index (FAISS, Chroma, pgvector) — `retrieve()` is
  the only interface the rest of the app depends on.
- CORS in `main.py` is locked to `localhost:5173` (Vite's default port) — update
  it for your deployed origin.
