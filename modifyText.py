from extractText import get_all_texts
import re
from datasets import Dataset
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token  # Critical for padding

def preprocess_document(text):
    """Clean document while preserving meaningful structure"""
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines
    text = re.sub(r'[ \t]+', ' ', text)          # Collapse multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)       # Limit consecutive newlines
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into properly sized chunks with overlap (in tokens)"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if not tokens:
            continue
            
        if len(tokens) > chunk_size:
            words = sentence.split()
            half = len(words) // 2
            first_half = ' '.join(words[:half])
            second_half = ' '.join(words[half:])
            sentences.extend([first_half, second_half])
            continue
            
        if current_length + len(tokens) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = current_chunk[-overlap:] + tokens
                current_length = len(current_chunk)
        else:
            current_chunk.extend(tokens)
            current_length += len(tokens)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def create_training_dataset(chunks):
    """Convert token chunks to properly formatted dataset with torch tensors"""
    input_ids = []
    attention_masks = []
    
    for chunk in chunks:
        encoded = tokenizer.prepare_for_model(
            chunk,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
    
    # Convert to tensors first
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = input_ids.clone()  # Proper tensor cloning
    
    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }).with_format("torch")

# Pipeline execution
raw_text = get_all_texts()
clean_text = preprocess_document(raw_text)
token_chunks = chunk_text(clean_text)
tokenized_dataset = create_training_dataset(token_chunks)