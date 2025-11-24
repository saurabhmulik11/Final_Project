import pdfplumber
import pytesseract
from PIL import Image


def extract_text_from_pdf(path):
    """Extract text from a PDF. Uses pdfplumber for text-backed PDFs and pytesseract as OCR fallback."""
    text_pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = None
            if text and text.strip():
                text_pages.append(text)
            else:
                try:
                    pil = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(pil)
                    text_pages.append(ocr_text)
                except Exception:
                    continue
    return "\n".join(text_pages)

# from PyPDF2 import PdfReader

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         reader = PdfReader(pdf_path)
#         for page in reader.pages:
#             text += page.extract_text() or ""
#     except Exception as e:
#         print(f"PDF extraction error: {e}")
    # return text
