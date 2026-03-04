export BRIDGE_API_KEY="super-secret-key"
export NIM_BASE_URL="http://127.0.0.1:8000/v1"
export NIM_MODEL="nvidia/cosmos-reason2-8b"
export PUBLIC_BASE_URL="http://host.docker.internal:8080"
python3 -m uvicorn cosmos_app:app --host 0.0.0.0 --port 8080
