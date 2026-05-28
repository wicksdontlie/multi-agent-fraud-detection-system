import os
import fitz
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DOCUMENTS_PATH = "documents"

all_text = []

def extract_text_from_pdf(pdf_path):

    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            page_text = page.get_text()

            # If PDF already contains text
            if page_text.strip():
                text += page_text

        doc.close()

    except Exception as e:
        print(f"Normal extraction failed: {e}")

    # OCR fallback
    if len(text.strip()) < 100:

        print(f"OCR Running on {pdf_path}")

        images = convert_from_path(pdf_path)

        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text

    return text


for file in os.listdir(DOCUMENTS_PATH):

    if file.endswith(".pdf"):

        path = os.path.join(DOCUMENTS_PATH, file)

        print(f"Processing: {file}")

        extracted_text = extract_text_from_pdf(path)

        all_text.append(extracted_text)

combined_text = "\n".join(all_text)

with open("combined_fraud_docs.txt", "w", encoding="utf-8") as f:
    f.write(combined_text)

print("All PDFs processed successfully.")