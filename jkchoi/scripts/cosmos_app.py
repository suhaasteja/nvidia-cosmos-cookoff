
import base64
import hmac
import json
import os
import time
import secrets
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="Cosmos Reason2 API Bridge", version="1.0")

# Load configuration from environment variables
API_KEY = os.environ.get("BRIDGE_API_KEY", "")
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/cosmos-reason2-2b")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/")

# Limit maximum accepted image size (bytes)
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(2 * 1024 * 1024)))

# Temporary image TTL (seconds)
IMAGE_TTL_S = int(os.environ.get("IMAGE_TTL_S", "120"))

# In-memory store for temporary images: token -> (bytes, mime, expires_at)
# _IMAGE_STORE: Dict[str, Tuple[bytes, str, float]] = {}
# token -> (bytes, mime, ext, expires_at)
_IMAGE_STORE: Dict[str, Tuple[bytes, str, str, float]] = {}

class Reason2Request(BaseModel):
    # Publicly accessible image URL (preferred)
    image_url: Optional[str] = Field(default=None, description="Public image URL.")

    # Base64 encoded image bytes (JPEG/PNG)
    image_b64: Optional[str] = Field(default=None, description="Base64 encoded image bytes.")

    # Natural language instruction for the model
    instruction: str = Field(..., min_length=1, max_length=2000)


def mime_to_ext(mime: str) -> str:
    # Map MIME type to file extension
    if mime == "image/jpeg":
        return "jpg"
    if mime == "image/png":
        return "png"
    return "bin"

def validate_key(x_api_key: str):
    # Use constant-time comparison for security
    if not API_KEY or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def guess_mime(image_bytes: bytes) -> str:
    # Detect MIME type using magic bytes
    if image_bytes.startswith(b"\xFF\xD8\xFF"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return "application/octet-stream"


def cleanup_expired_images():
    # Remove expired images from in-memory store
    now = time.time()
    expired = [k for k, (_, _, _, exp) in _IMAGE_STORE.items() if exp <= now]
    for k in expired:
        _IMAGE_STORE.pop(k, None)

def store_temp_image(image_bytes: bytes, mime: str) -> Tuple[str, str]:
    # Store image bytes temporarily and return (token, ext)
    cleanup_expired_images()
    token = secrets.token_urlsafe(24)
    ext = mime_to_ext(mime)
    expires_at = time.time() + IMAGE_TTL_S
    _IMAGE_STORE[token] = (image_bytes, mime, ext, expires_at)
    return token, ext


def extract_json(text: str) -> Dict[str, Any]:
    # Extract JSON object from model output safely
    s = text.strip()
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    return json.loads(s)


@app.get("/v1/images/{token}.{ext}")
def get_temp_image(token: str, ext: str):
    # Serve a temporarily stored image by token
    cleanup_expired_images()
    item = _IMAGE_STORE.get(token)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found or expired")

    image_bytes, mime, stored_ext, expires_at = item

    # Basic extension check to avoid confusion
    if ext.lower() != stored_ext.lower():
        raise HTTPException(status_code=404, detail="Image not found")

    headers = {"Cache-Control": "no-store", "Expires": "0"}
    return Response(content=image_bytes, media_type=mime, headers=headers)

@app.post("/v1/reason2/action")
def reason2_action(req: Reason2Request, x_api_key: str = Header(default="")):
    validate_key(x_api_key)

    instruction = req.instruction.strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction must be non-empty")

    # Decide which image input URL to send to NIM
    image_input_url: Optional[str] = None

    if req.image_url:
        # Use the provided public URL directly
        image_input_url = req.image_url.strip()

    elif req.image_b64:
        # Decode base64 image and host it temporarily as an HTTP URL
        try:
            image_bytes = base64.b64decode(req.image_b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image_b64 (not valid base64)")

        if not image_bytes:
            raise HTTPException(status_code=400, detail="image_b64 decoded to empty bytes")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")

        mime = guess_mime(image_bytes)
        #token = store_temp_image(image_bytes, mime)
        #image_input_url = f"{PUBLIC_BASE_URL}/v1/images/{token}"

        token, ext = store_temp_image(image_bytes, mime)
        image_input_url = f"{PUBLIC_BASE_URL}/v1/images/{token}.{ext}"

    else:
        raise HTTPException(status_code=400, detail="Either image_url or image_b64 must be provided")

    # Enforce strict JSON output from the model
    #prompt = (
    #    "You are a robot vacuum front-facing camera.\n"
    #    "Decide if the forward path is BLOCKED.\n"
    #    "Definition of BLOCKED (conservative):\n"
    #    "- If any solid obstacle (cone, barrier, box, wall, furniture) is visible in the LOWER HALF of the image,\n"
    #    "  assume it may block the robot within ~1 meter and set blocked=true.\n"
    #    "- Otherwise set blocked=false.\n"
    #    "Return ONLY valid JSON:\n"
    #    "{\n"
    #    '  "reasoning": "one short sentence",\n'
    #    '  "blocked": true or false\n'
    #    "}\n"
    #    f"Instruction: {instruction}"
    #)

    #prompt = (
    #    "You are the real-time visual-spatial reasoning engine for a robot vacuum. You operate in a continuous loop, receiving a front-facing camera image and current odometry."
    #    "Your Mission: Navigate the space to clean efficiently while strictly avoiding collisions with objects, walls, or drops."
    #    "Capabilities & Rules:"
    #    "You can only move by turning a relative number of degrees and driving a relative distance in meters."
    #    "turn_degrees: Positive numbers turn RIGHT. Negative numbers turn LEFT. 0 means keep the steering straight. Max turn per step is 90 degrees."
    #    "move_meters: Positive numbers drive FORWARD. Negative numbers drive BACKWARD. 0 means stop. Max drive per step is 0.5 meters."
    #    "Safety First: If an obstacle is directly in front of you (less than 0.3m), you MUST stop (move_meters: 0) or back up, and turn to clear it."
    #    "Reasoning: Always analyze the scene in scene_analysis before outputting movement commands."
    #    "You MUST respond ONLY with a raw, valid JSON object matching this exact schema, with no markdown formatting or conversational text:"
    #    "{"
    #    '   "scene_analysis": "Briefly describe the floor ahead, objects, and safe paths.",'
    #    '   "obstacle_detected": true/false,'
    #    '   "turn_degrees": 0.0,'
    #    '   "move_meters": 0.0'
    #    "}"
    #    f"Instruction: {instruction}"
    #)

    #payload = {
    #    "model": NIM_MODEL,
    #    "messages": [
    #        {"role": "system", "content": "You are a robot navigation assistant."},
    #        {
    #            "role": "user",
    #            "content": [
    #                {"type": "image_url", "image_url": {"url": image_input_url}},
    #                {"type": "text", "text": prompt},
    #            ],
    #        },
    #    ],
    #    "max_tokens": 300,
    #    "stream": False,
    #}

    ## Call NIM and surface the error body for debugging
    #r = requests.post(f"{NIM_BASE_URL}/chat/completions", json=payload, timeout=30)
    #if r.status_code >= 400:
    #    raise HTTPException(status_code=502, detail=f"NIM error {r.status_code}: {r.text}")

    #model_text = r.json()["choices"][0]["message"]["content"]

    ## Parse model output JSON
    #try:
    #    parsed = extract_json(model_text)
    #    reasoning = str(parsed.get("reasoning", ""))
    #    blocked = bool(parsed.get("blocked", False))
    #except Exception:
    #    raise HTTPException(status_code=500, detail="Model output was not valid JSON")
    #
    ## Map decision to action deterministically (server-side policy)
    #action = {"move": "stop", "distance": 0.0} if blocked else {"move": "linear_x", "distance": 0.5}
 
   
    #return {"reasoning": reasoning, "action": action}

    # Build strict navigation decision prompt
    prompt = (
        "You are a robot vacuum front-facing camera navigation system.\n"
        "Analyze the scene and decide the next immediate safe movement.\n"
        "Rules:\n"
        "- If any obstacle is within ~0.5 meters in front, obstacle_detected=true.\n"
        "- If obstacle_detected=true:\n"
        "  * move_meters must be 0.0\n"
        "  * choose turn_degrees between -90 and 90 (negative=left, positive=right)\n"
        "- If no obstacle:\n"
        "  * obstacle_detected=false\n"
        "  * move_meters between 0.2 and 0.8\n"
        "  * turn_degrees must be 0.0\n"
        "Return ONLY valid JSON with this structure:\n"
        "{\n"
        '  "scene_analysis": "one short paragraph",\n'
        '  "obstacle_detected": true or false,\n'
        '  "turn_degrees": number,\n'
        '  "move_meters": number\n'
        "}\n"
        f"{instruction}"
    )
    
    payload = {
        "model": NIM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise robotics navigation assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_input_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": 400,
        "stream": False,
    }
    
    r = requests.post(f"{NIM_BASE_URL}/chat/completions", json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"NIM error {r.status_code}: {r.text}")
    
    model_text = r.json()["choices"][0]["message"]["content"]
    
    # Parse model output JSON
    try:
        parsed = extract_json(model_text)
    
        scene_analysis = str(parsed.get("scene_analysis", ""))
        obstacle_detected = bool(parsed.get("obstacle_detected", False))
        turn_degrees = float(parsed.get("turn_degrees", 0.0))
        move_meters = float(parsed.get("move_meters", 0.0))
    
    except Exception:
        raise HTTPException(status_code=500, detail="Model output was not valid JSON")
    
    # Safety normalization (server-side policy enforcement)
    
    if obstacle_detected:
        move_meters = 0.0
        # Clamp turn range
        turn_degrees = max(min(turn_degrees, 90.0), -90.0)
    else:
        turn_degrees = 0.0
        move_meters = max(min(move_meters, 0.8), 0.2)
    
    return {
        "scene_analysis": scene_analysis,
        "obstacle_detected": obstacle_detected,
        "turn_degrees": turn_degrees,
        "move_meters": move_meters
    }


