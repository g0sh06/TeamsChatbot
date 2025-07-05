from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

app = Flask(__name__)

print("Loading model")
model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)
peft_model_path = os.path.abspath("my_tinyllama_finetuned")
model = PeftModel.from_pretrained(
    base_model, 
    peft_model_path
).to("cpu")
print("Model loaded successfully!")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('text', '')
    
    if not user_message:
        return jsonify({"error" : "No text provided"}), 400
    
    # Generate response
    inputs = tokenizer(user_message, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
    )
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # Clean up response
    response = response.replace(user_message, "").strip()

    # Return the response 
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)