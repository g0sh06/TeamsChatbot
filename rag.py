from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import re

class RAGSystem:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32
        ).to(self.device)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_chunks = []
        self.document_embeddings = None
        
    def chunk_document(self, text, chunk_size=500):
        sections = re.split(r'\n\s*\n', text)
        chunks = []
        for section in sections:
            words = section.split()
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                chunks.append(chunk)
        return chunks
    
    def index_documents(self, documents):
        self.document_chunks = []
        for doc in documents:
            self.document_chunks.extend(self.chunk_document(doc))
        self.document_embeddings = self.embedding_model.encode(self.document_chunks)
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.document_chunks[i] for i in top_indices]
    
    def generate_response(self, question, context):
        prompt = f"""You are a helpful assistant for the Data Science and AI course at HZ University.
Use the context below to answer the user's question.
If the answer is not in the context, say: "I couldn't find that information in the course documents."

Context:
{context}

Question:
{question}

Answer:"""

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the generated text (after the prompt)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def answer_question(self, question):
        """End-to-end RAG pipeline with proper error handling"""
        if not self.document_chunks:
            return "I couldn't find any course documents to reference. Please ensure documents are properly loaded."
        
        try:
            relevant_chunks = self.retrieve_relevant_chunks(question)
            context = "\n\n".join(relevant_chunks)
            return self.generate_response(question, context)
        except Exception as e:
            return f"Error processing your question: {str(e)}"