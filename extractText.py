import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
import shutil
from langchain_chroma import Chroma

load_dotenv()
DATA_PATH = os.getenv("FOLDER")
CHROMA_PATH = "database"

gpt4all_embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    device="cpu",  # Force CPU mode to avoid DLL errors
    gpt4all_kwargs={'allow_download': True}  # Removed quotes from True
)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
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
    
