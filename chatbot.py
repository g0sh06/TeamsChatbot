from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True  
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    low_cpu_mem_usage=True, 
    attn_implementation="sdpa"  
)

lora_config = LoraConfig(
    r=4,  
    lora_alpha=16,  
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, 
    task_type=TaskType.CAUSAL_LM,
    bias="none"  
)

model = get_peft_model(model, lora_config)
