import os
from dotenv import load_dotenv
import pdfplumber
from docx import Document

# Load environment variables
load_dotenv()
folder = os.getenv("FOLDER")

if not folder:
    print("FOLDER not set in .env")
    exit()

folder = os.path.normpath(folder)

def extract_text_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                # Extract plain text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

                # Extract and convert tables
                tables = page.extract_tables()
                for table in tables:
                    headers = table[0]
                    for row in table[1:]:
                        # Skip if all cells are empty
                        if not any(cell for cell in row):
                            continue
                        sentence_parts = []
                        for header, cell in zip(headers, row):
                            if header and cell:
                                sentence_parts.append(f"{header.strip()}: {cell.strip()}")
                        if sentence_parts:
                            text += "\n" + ". ".join(sentence_parts) + ".\n"

    except Exception as e:
        print(f"Failed to extract PDF ({path}): {e}")
    return text.strip()

def extract_text_docx(path):
    try:
        doc = Document(path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Failed to extract DOCX ({path}): {e}")
        return ""

def get_all_texts(folder_path=None):
    folder_path = folder_path or folder
    texts = []

    for file in os.listdir(folder_path):
        path = os.path.normpath(os.path.join(folder_path, file))

        if os.path.isfile(path) and not file.startswith('.') and not file.endswith(".ini"):
            print(f"\nReading: {file}")
            text = ""

            if file.lower().endswith(".pdf"):
                text = extract_text_pdf(path)
            elif file.lower().endswith(".docx"):
                text = extract_text_docx(path)
            else:
                print(f"Skipping unsupported file type: {file}")
                continue

            if text:
                texts.append(text)
            else:
                print(f"No extractable content in {file}")
    
    return "\n\n".join(texts)
