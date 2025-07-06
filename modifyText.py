from extractText import get_all_texts
import re
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import Dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def preprocess_document(text):
    """Clean document while preserving meaningful breaks"""
    # Normalize whitespace but keep paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines
    text = re.sub(r'[ \t]+', ' ', text)          # Collapse multiple spaces
    return text.strip()

def chunk_text(text, chunk_size=512, overlap=64):
    """Split text into properly sized chunks with overlap"""
    # First split into sentences (preserving punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Tokenize sentence without special tokens (we'll add them later)
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(tokens)
        
        # If adding this sentence would exceed chunk size
        if current_length + sentence_length > chunk_size:
            if current_chunk:  # Save current chunk if not empty
                chunks.append(current_chunk)
            
            # Start new chunk with overlap from previous
            overlap_size = min(overlap, len(current_chunk))
            current_chunk = current_chunk[-overlap_size:] + tokens
            current_length = len(current_chunk)
        else:
            current_chunk.extend(tokens)
            current_length += sentence_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def create_training_dataset(chunks):
    """Convert token chunks to properly formatted dataset"""
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for chunk in chunks:
        # Prepare model inputs with special tokens
        inputs = tokenizer.prepare_for_model(
            chunk,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        dataset_dict["input_ids"].append(inputs["input_ids"].squeeze(0))
        dataset_dict["attention_mask"].append(inputs["attention_mask"].squeeze(0))
        dataset_dict["labels"].append(inputs["input_ids"].squeeze(0).clone())
    
    return Dataset.from_dict(dataset_dict).with_format("torch")

raw_text = get_all_texts()
clean_text = preprocess_document(raw_text)
chunks = chunk_text(clean_text)
tokenized_dataset = create_training_dataset(chunks)