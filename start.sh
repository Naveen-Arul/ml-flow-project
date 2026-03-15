#!/bin/bash
# Start FastAPI backend on internal port 8000
gunicorn -w 1 -k uvicorn.workers.UvicornWorker backend:app --bind 0.0.0.0:8000 &
# Wait 10 seconds for backend initialization
sleep 10
# Start Streamlit Frontend on the specific port Railway provides
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
