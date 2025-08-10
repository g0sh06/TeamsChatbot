FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0

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

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Copy all app files (including start.sh if you have one)
COPY . .

# Create startup script directly in final location
RUN echo '#!/bin/bash\n\
ollama serve > /var/log/ollama.log 2>&1 & \n\
sleep 15 \n\
ollama pull mistral \n\
streamlit run main.py --server.port=8501 --server.address=0.0.0.0' > /start.sh && \
    chmod +x /start.sh

EXPOSE 8501 11434

CMD ["/bin/bash", "/start.sh"]
