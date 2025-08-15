#!/bin/bash

# Start Ollama in the background
ollama serve > /var/log/ollama.log 2>&1 &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434 >/dev/null; do
  sleep 1
done

# Pull the model (may take time)
echo "Pulling mistral model..."
ollama pull mistral

# Start Streamlit (this keeps the container running)
echo "Starting Streamlit server..."
exec streamlit run main.py --server.port=8501 --server.address=0.0.0.0