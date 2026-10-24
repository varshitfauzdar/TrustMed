from io import BytesIO


def extract_pdf_text(pdf_bytes):
    """Extract text from a PDF using PyMuPDF; no OCR is performed."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs PyMuPDF. Install it with: "
            r".\.venv\Scripts\python.exe -m pip install pymupdf"
        ) from exc

    document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        pages = []
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[Page {page_number}]\n{text}")
        extracted = "\n\n".join(pages).strip()
        if not extracted:
            return (
                "No selectable text was found in this PDF. It may be scanned; "
                "this prototype does not perform OCR."
            )
        return extracted
    finally:
        document.close()
