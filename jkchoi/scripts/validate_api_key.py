import os
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

API_KEY = os.environ.get("BRIDGE_API_KEY")

def validate_api_key(x_api_key: str):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

