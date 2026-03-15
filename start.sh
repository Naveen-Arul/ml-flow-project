#!/bin/bash
# 1. Kill any existing processes (safety)
pkill -f gunicorn || true
pkill -f streamlit || true

# 2. Start FastAPI backend on internal port 5000
# Using -b 0.0.0.0:5000 to be absolutely sure
gunicorn -w 1 -k uvicorn.workers.UvicornWorker backend:app --bind 0.0.0.0:5000 --daemon

# 3. Wait for backend
sleep 15

# 4. Start Streamlit Frontend
# Streamlit MUST use $PORT provided by Railway
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
