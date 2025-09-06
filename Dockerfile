FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_NO_CUDA=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add to your Dockerfile
RUN wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz && \
    tar -xzf ngrok-v3-stable-linux-amd64.tgz && \
    mv ngrok /usr/local/bin/ && \
    rm ngrok-v3-stable-linux-amd64.tgz
    
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Create necessary directories
RUN mkdir -p /app/uploaded_docs /app/database

# Copy application files
COPY . .

# Set up startup script
RUN echo '#!/bin/bash\n\
ollama serve > /var/log/ollama.log 2>&1 &\n\
sleep 15\n\
ollama pull mistral\n\
streamlit run main.py --server.port=8501 --server.address=0.0.0.0' > /start.sh && \
    chmod +x /start.sh

EXPOSE 8501 11434

CMD ["/bin/bash", "/start.sh"]