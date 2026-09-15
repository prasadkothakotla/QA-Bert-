"""
Plain-text extraction from PDFs, used by the /documents/upload route.

This handles text-based PDFs. Scanned/image PDFs have no embedded text layer
and will come back empty or near-empty -- if you need those, run OCR
(e.g. pytesseract) on the page images before extraction and feed the result
into `extract_text_from_pdf` the same way.
"""
from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(raw_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(raw_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages).strip()
