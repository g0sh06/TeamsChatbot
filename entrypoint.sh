#!/bin/bash

ollama serve &

ollama pull mistral

streamlit run main.py