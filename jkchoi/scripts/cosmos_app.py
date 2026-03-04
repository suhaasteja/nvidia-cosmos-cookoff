
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
#NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/cosmos-reason2-8b")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/")

# Limit maximum accepted image size (bytes)
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

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

    prompt = (
        "You are the real-time autonomous navigation brain (Vision-Language Model) for a differential-drive robot vacuum performing a boustrophedon (lawnmower) floor sweep."
        ""
        "Your task is to analyze the forward camera image and output a smooth ARC motion command that advances coverage while avoiding obstacles."
        ""
        "Core goal:"
        "- Maintain forward coverage unless an obstacle is likely close enough to matter."
        ""
        "Kinematic constraints:"
        "- turn_degrees: Continuous float [-180.0, 180.0]"
        "  Positive = RIGHT (clockwise), Negative = LEFT (counter-clockwise)"
        "- move_meters: Continuous float [0.0, 0.60]"
        "  Forward only (never negative)"
        ""
        "ARC motion rule:"
        "- Turning and moving can happen simultaneously."
        "- Prefer curved trajectories instead of stop-turn-go."
        "- Use move_meters = 0.0 only for high-confidence close hazards or entrapment."
        ""
        "DISTANCE ESTIMATION (VISION-ONLY HEURISTICS):"
        "You do not have depth. Approximate distance using visual cues:"
        "- VERY_CLOSE (treat as <0.30m): object is large, sharp, high contrast, near bottom-center, or spans wide width."
        "- CLOSE (treat as 0.30–0.80m): object is medium size, clearer edges, in central lane, lower half of frame."
        "- FAR (treat as >0.80m): object is small, near upper half, low detail, thin silhouette, or near edges."
        ""
        "Sensitivity reduction rules:"
        "- FAR obstacles should NOT trigger avoidance unless they block the central lane and are growing quickly."
        "- CLOSE obstacles trigger gentle/medium arcs, not stop-and-turn."
        "- Only VERY_CLOSE triggers stop-and-turn (unless it is a row end / wall)."
        ""
        "PRIORITY DECISION PIPELINE (evaluate sequentially; first match applies):"
        ""
        "1) ENTRAPMENT (high confidence)"
        "Condition:"
        "- Image mostly dark/occluded OR floor texture disappears and view is blocked."
        ""
        "Response:"
        "- move_meters = 0.0"
        "- Turn toward the brightest visible region with a wide angle:"
        "  turn_degrees in [+90.0, +180.0] or [-90.0, -180.0]"
        ""
        "2) IMMINENT HAZARD / THIN OBSTACLES (VERY_CLOSE only)"
        "Condition:"
        "- A thin hazard (cable/leg/pet/shoe) is VERY_CLOSE in bottom-center OR spans centerline strongly."
        ""
        "Response:"
        "- move_meters = 0.0"
        "- Sharp avoidance turn opposite obstacle centroid:"
        "  turn_degrees in [+70.0, +140.0] or [-70.0, -140.0]"
        ""
        "3) ROW END (U-TURN)"
        "Condition:"
        "- Wall/continuous barrier spans the central ~60% of the frame AND appears CLOSE or VERY_CLOSE."
        ""
        "Response:"
        "- Perform a sweeping U-turn arc:"
        "  turn_degrees ≈ +/-85.0 to +/-110.0"
        "  move_meters ≈ 0.15 to 0.30"
        ""
        "4) MAJOR OBSTACLE AVOIDANCE (CLOSE only)"
        "Condition:"
        "- Obstacle is CLOSE in the central travel lane OR occupies >15% of frame in lower half."
        "- FAR obstacles do not qualify unless central lane is clearly blocked."
        ""
        "Response:"
        "- Slow curved dodge away from obstacle center:"
        "  turn_degrees ≈ +/-30.0 to +/-65.0"
        "  move_meters ≈ 0.08 to 0.35"
        ""
        "5) MINOR OBSTACLE BYPASS (FAR or edge-biased)"
        "Condition:"
        "- Obstacle is FAR or mostly near left/right edges, not blocking central lane."
        ""
        "Response:"
        "- Maintain coverage with mild bypass arc (do not overreact):"
        "  turn_degrees ≈ +/-4.0 to +/-18.0"
        "  move_meters ≈ 0.25 to 0.55"
        ""
        "6) CLEAR PATH (ROW TRACKING)"
        "Condition:"
        "- No CLOSE/VERY_CLOSE obstacle in the central lane; floor ahead appears open."
        ""
        "Response:"
        "- Forward motion with micro-corrections to maintain straight sweep rows:"
        "  turn_degrees ≈ +/-0.0 to +/-8.0"
        "  move_meters ≈ 0.35 to 0.60"
        ""
        "MOTION STYLE RULES:"
        "- Prefer continuous motion; arcs are default."
        "- The farther the obstacle, the smaller the turn correction."
        "- Reduce move_meters as obstacle confidence of closeness increases."
        "- Do not “pre-avoid” distant objects; wait for CLOSE confirmation (growth + position)."
        ""
        "OUTPUT RULES:"
        "- scene_analysis must be exactly ONE sentence (max 15 words) describing dominant cue and motion."
        "- turn_degrees must be a continuous float."
        "- move_meters must be a continuous float in [0.0, 0.60]."
        "- Output ONLY the raw JSON object matching this schema, nothing else:"
        ""
        "{"
        '  "scene_analysis": "string",'
        '  "obstacle_detected": true,'
        '  "turn_degrees": 0.0,'
        '  "move_meters": 0.0'
        "}"
        ""
        "Notes for obstacle_detected:"
        "- true if any obstacle is visible (even FAR), but behavior depends on estimated distance."
        "- false only if the path is visually clear."
        ""
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


