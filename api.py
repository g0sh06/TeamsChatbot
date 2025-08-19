from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import requests  # Add this import

app = Flask(__name__)

print("Loading model")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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

# Add function to query Ollama for RAG context
def get_rag_context(query):
    """Get context from Ollama RAG service"""
    try:
        # Use the Ollama service for embeddings and retrieval
        ollama_url = f"http://rag-chatbot:11434"  # Internal Docker network
        # For simplicity, we'll use the local model for now
        # In production, you'd call your RAG service here
        return f"Context for: {query}"
    except Exception as e:
        print(f"Error getting RAG context: {e}")
        return ""

def format_for_teams(response_text):
    """Format response for Microsoft Teams"""
    return {
        "type": "message",
        "text": response_text
    }

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return jsonify({"status": "OK"}), 200
        
    data = request.json
    
    # Handle different message formats
    if 'text' in data:
        user_message = data.get('text', '')
    elif 'value' in data and 'text' in data['value']:
        user_message = data['value']['text']
    else:
        return jsonify({"error": "No text provided"}), 400
    
    if not user_message:
        return jsonify({"error": "No text provided"}), 400
    
    # Get RAG context (in a real implementation)
    context = get_rag_context(user_message)
    
    # Generate response
    prompt = f"Context: {context}\n\nQuestion: {user_message}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    # Format for Teams
    teams_response = format_for_teams(response)
    
    return jsonify(teams_response)

@app.route('/api/messages', methods=['POST'])
def bot_framework_messages():
    """Endpoint for Azure Bot Service"""
    activity = request.json
    if activity['type'] == 'message':
        user_message = activity['text']
        
        # Generate response
        inputs = tokenizer(user_message, return_tensors="pt").to("cpu")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(user_message, "").strip()
        
        # Return proper Bot Framework response
        return jsonify({
            "type": "message",
            "text": response
        })
    
    return jsonify({"status": "ok"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)