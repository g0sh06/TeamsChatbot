
from langchain_huggingface import HuggingFaceEmbeddings
gpt4all_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
