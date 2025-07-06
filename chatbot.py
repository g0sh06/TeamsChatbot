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
    torch_dtype=torch.float32,  
    device_map="cpu",           
    low_cpu_mem_usage=True      
)

lora_config = LoraConfig(
    r=8,  # Increased rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="lora_only"
)

model = get_peft_model(model, lora_config)
