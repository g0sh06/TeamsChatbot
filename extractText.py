import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
import shutil
from langchain_chroma import Chroma

load_dotenv()
DATA_PATH = os.getenv("FOLDER")
CHROMA_PATH = "database"

gpt4all_embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': 'True'}
)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents=documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, gpt4all_embeddings, persist_directory=CHROMA_PATH
    )    

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

def create_vector_db():
    "Create vector DB from personal PDF files."
    documents = load_documents()
    doc_chunks = split_text(documents)
    save_to_chroma(doc_chunks)

if __name__ == "__main__":    
    create_vector_db()

    
