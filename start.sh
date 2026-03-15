#!/bin/bash
# 1. Train models if they don't exist (ensures Railway has models to serve)
if [ ! -f "models/logistic_regression.pkl" ]; then
    echo "No models found. Training models..."
    python news_classification.py
fi

# 2. Start the FastAPI backend in the background
echo "Starting FastAPI Backend..."
gunicorn -w 1 -k uvicorn.workers.UvicornWorker backend:app --bind 0.0.0.0:8000 &

# 3. Start the Streamlit frontend
echo "Starting Streamlit Frontend on port $PORT..."
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
