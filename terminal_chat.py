from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Loading the model
model_name = "my_tinyllama_finetuned"
# Added tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32,
    device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, "my_tinyllama_finetuned").to("cpu")

SYSTEM_PROMPT = "You are a helpful assistant for a Data Science and AI (DSAI) course. Provide concise, accurate answers to questions about the course."

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
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.2, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.split("<|assistant|>")[-1].strip()
        print("\nAI:", assistant_response)

if __name__ == "__main__":
    chat()
    print("Chat session ended.")