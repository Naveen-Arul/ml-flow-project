#!/bin/bash
# Start FastAPI backend on internal port 5000 (Avoid 8000/8080 completely)
gunicorn -w 1 -k uvicorn.workers.UvicornWorker backend:app --bind 0.0.0.0:5000 &
# Wait 10 seconds for backend initialization
sleep 10
# Start Streamlit Frontend on the specific port Railway provides
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
