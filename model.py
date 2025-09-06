from langchain_community.embeddings import HuggingFaceEmbeddings

gpt4all_embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)