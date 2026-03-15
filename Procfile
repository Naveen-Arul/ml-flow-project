web: gunicorn -w 1 -k uvicorn.workers.UvicornWorker backend:app & streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
