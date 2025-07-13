from langchain_community.embeddings import HuggingFaceEmbeddings

qwen_embedding_model_name = "Qwen/Qwen3-Embedding-8B"

gpt4all_embeddings = HuggingFaceEmbeddings(
    model_name=qwen_embedding_model_name,
    model_kwargs={"device": "cpu"},  # or "cpu" if you don't have GPU
    encode_kwargs={"normalize_embeddings": True}
)