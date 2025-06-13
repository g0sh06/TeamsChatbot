import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from docx import Document

# Load environment variables
load_dotenv()
folder = os.getenv("FOLDER")

if not folder:
    print("⚠️ FOLDER not set in .env")
    exit()

folder = os.path.normpath(folder)

def extract_text_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_docx(path):
    doc = Document(path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs)

for file in os.listdir(folder):
    path = os.path.normpath(os.path.join(folder, file))

    if os.path.isfile(path) and not file.startswith('.') and not file.endswith(".ini"):
        print(f"\nReading: {file}")

        try:
            if file.lower().endswith(".pdf"):
                text = extract_text_pdf(path)
            elif file.lower().endswith(".docx"):
                text = extract_text_docx(path)
            else:
                print(f"Skipping unsupported file type: {file}")
                continue

            print(text)  # Show first 300 chars of extracted text

        except Exception as e:
            print(f"Error reading {file}: {e}")
