import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from model import gpt4all_embeddings
import tiktoken
import shutil
import time


load_dotenv()
CHROMA_PATH = os.path.abspath("database")  # Use absolute path
os.makedirs(CHROMA_PATH, exist_ok=True)  # Ensure directory exists

def tiktoken_len(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # Used by OpenAI's GPT-4
    return len(encoding.encode(text))

def create_vector_db_from_files(file_paths: list):
    """Create vector DB from specific file paths"""
    print(f"\nStarting processing for {len(file_paths)} files...")
    documents = load_documents_from_files(file_paths)
    doc_chunks = split_text(documents)
    save_to_chroma(doc_chunks)

def load_documents_from_files(file_paths: list):
    """Load documents from specific file paths"""
    documents = []
    for file_path in file_paths:
        try:
            print(f"\nLoading: {file_path}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Successfully loaded {len(docs)} pages")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
    return documents

def split_text(documents: list[Document]):
    print("\nSplitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    os.makedirs(CHROMA_PATH, exist_ok=True)

# Clear only the contents, not the mount point
    for filename in os.listdir(CHROMA_PATH):
        file_path = os.path.join(CHROMA_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"⚠ Failed to delete {file_path}: {e}")

    Chroma.from_documents(
        chunks, 
        gpt4all_embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    # Example usage (replace with your actual file paths)
    test_files = [
        os.path.join(os.getenv("FOLDER", "uploaded_docs"), "DSAI_courseoutline.pdf")
    ]
    create_vector_db_from_files(test_files)