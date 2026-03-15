#!/bin/bash
# Start the FastAPI backend in the background
gunicorn -w 1 -k uvicorn.workers.UvicornWorker backend:app --bind 0.0.0.0:8000 &

# Start the Streamlit frontend
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
