from langchain_community.embeddings import HuggingFaceEmbeddings

gpt4all_embeddings = HuggingFaceEmbeddings(
     model_name="sentence-transformers/all-mpnet-base-v2"
)