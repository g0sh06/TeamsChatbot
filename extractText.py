from PyPDF2 import PdfReader
from docx import Document  # From python-docx package
import os

def get_pdf_text(pdf_path):
    """Extract text from PDF files"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def get_docx_text(docx_path):
    """Extract text from DOCX files"""
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
    return text

def get_all_texts(docs_folder="documents"):
    """Get text from all supported documents in folder"""
    texts = []
    for filename in os.listdir(docs_folder):
        path = os.path.join(docs_folder, filename)
        try:
            if filename.lower().endswith(".pdf"):
                texts.append(get_pdf_text(path))
            elif filename.lower().endswith(".docx"):
                texts.append(get_docx_text(path))
            # Add other file types as needed
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return texts