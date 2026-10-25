from io import BytesIO


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extract selectable text from a PDF page by page.

    Returns structured records so retrieval can preserve page-level
    evidence provenance. OCR is not performed.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs PyMuPDF. Install it with: "
            r"..\.venv\Scripts\python.exe -m pip install pymupdf"
        ) from exc

    document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")

    try:
        pages = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

        return pages

    finally:
        document.close()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Preserve the original text-extraction interface.

    Pages are labelled so their provenance remains visible in the text.
    """
    pages = extract_pdf_pages(pdf_bytes)

    if not pages:
        return (
            "No selectable text was found in this PDF. It may be scanned; "
            "this prototype does not perform OCR."
        )

    return "\n\n".join(
        f"[Page {page['page_number']}]\n{page['text']}"
        for page in pages
    )
