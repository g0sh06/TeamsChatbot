from extractText import get_all_texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from datasets import Dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Get raw text
raw_text = get_all_texts()  # Now returns a single string

def preprocess_text(text):
    """Clean and normalize the text"""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess the text
processed_text = preprocess_text(raw_text)

def chunk_text(text, chunk_size=200, overlap=30):
    tokens = tokenizer.tokenize(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = ' '.join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    
        if i + chunk_size >= len(tokens):
            break
            
    return chunks

text_chunks = chunk_text(processed_text)

text_dataset = Dataset.from_dict({"text": text_chunks})

def tokenize_function(examples):
    """Tokenize the text chunks"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=384,
        padding="max_length"  # Added for consistent length
    )