from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Loading the model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Added tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, "my_tinyllama_finetuned").to("cpu")


SYSTEM_PROMPT = """You are an expert assistant for the HZ University DSAI course. 
Answer ONLY using information from the official 2024-2025 course outline document.
If information isn't in the document, say "This information is not specified in the course outline."
For dates and schedules, refer specifically to the course schedule section.
"""

print("\nDSAI Course Assistant ready! Type 'quit' to exit.")

def chat():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        # Format with system prompt and user question
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cpu")

        outputs = model.generate(
            inputs,
            max_new_tokens=250,
            temperature=0.2,  # lower temeprature = more specific answer
            do_sample=True,
            top_p=0.9
        )
        
        # Clean the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.split("<|assistant|>")[-1].strip()
        print("\nAI:", assistant_response)

if __name__ == "__main__":
    chat()
    print("Chat session ended.")