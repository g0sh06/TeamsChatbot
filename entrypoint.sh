#!/bin/bash

# Start Ollama
ollama serve > /var/log/ollama.log 2>&1 &

# Wait for Ollama
until curl -s http://localhost:11434 >/dev/null; do
  sleep 1
done

# Pull model
ollama pull mistral

# Start ngrok tunnel
ngrok http 8501 > /var/log/ngrok.log 2>&1 &

# Get public URL (wait a bit for ngrok to start)
sleep 5
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')
echo "Public URL: $PUBLIC_URL"

# Start Streamlit
exec streamlit run main.py --server.port=8501 --server.address=0.0.0.0