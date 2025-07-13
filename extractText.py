import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
import shutil
from langchain_chroma import Chroma
from model import gpt4all_embeddings
import tiktoken

load_dotenv()
DATA_PATH = os.getenv("FOLDER")
CHROMA_PATH = "database"


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_kwargs={"add_page_number": True})
    documents = loader.load()
    return documents

def tiktoken_len(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # Used by OpenAI's GPT-4
    return len(encoding.encode(text))

def split_text(documents: list[Document]):
    text_splitter = NLTKTextSplitter()
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(
        chunks, 
        gpt4all_embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

def create_vector_db():
    documents = load_documents()
    doc_chunks = split_text(documents)
    save_to_chroma(doc_chunks)

if __name__ == "__main__":
    create_vector_db()
    
