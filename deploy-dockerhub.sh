#!/bin/bash

# Deploy RAG chatbot from Docker Hub
az container create \
  --resource-group ChatbotRG \
  --name rag-chatbot \
  --image yourdockerhubusername/rag-chatbot:latest \
  --cpu 2 \
  --memory 4 \
  --ports 11434 \
  --ip-address Public \
  --dns-name-label my-rag-chatbot \
  --environment-variables OLLAMA_HOST=0.0.0.0

# Deploy API service from Docker Hub
az container create \
  --resource-group ChatbotRG \
  --name chatbot-api \
  --image yourdockerhubusername/chatbot-api:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --ip-address Public \
  --dns-name-label my-chatbot-api \
  --environment-variables OLLAMA_HOST=my-rag-chatbot.eastus.azurecontainer.io