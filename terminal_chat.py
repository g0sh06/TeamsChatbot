from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

print("=== Starting Chatbot ===")  # Debug line

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading tokenizer from {model_name}...")  # Debug line
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading base model...")  # Debug line
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
except Exception as e:
    print(f"❌ Failed to load base model: {e}")
    exit()

peft_model_path = os.path.abspath("my_tinyllama_finetuned")
print(f"Looking for fine-tuned model at: {peft_model_path}")  # Debug line

if not os.path.exists(peft_model_path):
    print(f"❌ Directory not found: {peft_model_path}")
    print("Please run fine-tuning.py first to create the model")
    exit()

print("Loading fine-tuned adapter...")  # Debug line
try:
    model = PeftModel.from_pretrained(base_model, peft_model_path).to("cpu")
    print("✅ Model loaded successfully!")  # Debug line
except Exception as e:
    print(f"❌ Failed to load fine-tuned model: {e}")
    print("Required files in the directory:")
    print("- adapter_config.json")
    print("- adapter_model.bin")
    exit()

def chat():
    print("\nChatbot ready! Type 'quit' to exit.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            print("Generating response...")  # Debug line
            inputs = tokenizer(user_input, return_tensors="pt").to("cpu")
            outputs = model.generate(**inputs, max_new_tokens=100)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("\nAI:", response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            continue

if __name__ == "__main__":
    chat()
    print("Chat session ended.")