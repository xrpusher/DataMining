# How to run:

1) run train_model.py
2) run uvicorn inference_server:app --host 127.0.0.1 --port 8001
3) run uvicorn backend_server:app --host 127.0.0.1 --port 8000
4) run client.py
