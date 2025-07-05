from extractText import get_all_texts
import re
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size=512, overlap=30):
    inputs = tokenizer(
        text,
        truncation=False,
        return_overflowing_tokens=True,
        stride=overlap,
        max_length=chunk_size,
        return_tensors="pt"
    )
    
    chunks = [tokenizer.decode(chunk, skip_special_tokens=True) 
             for chunk in inputs["input_ids"]]
    
    return chunks

raw_text = get_all_texts()
processed_text = preprocess_text(raw_text)

text_chunks = chunk_text(processed_text, chunk_size=512, overlap=30)

text_dataset = Dataset.from_dict({"text": text_chunks})