#!/usr/bin/env python3
import os
import sys
import json
import base64
import argparse
from io import BytesIO

import requests
from PIL import Image


#DEFAULT_IMAGE_URL = "https://media.discordapp.net/attachments/1469185959500316873/1478445984022265856/AOI_d_86w705b7ruYIAfJ5cA1vJ9UTe0RKT_SIzEYGbiiHcqw-DIt2dBLkb_4sm49oHPc11Ys05npaNJOq-1ldaJP_6G1odcOFKYcIhy6PgPQkxMcSbYVsm4amVb5GPNHoDMAOZoGnY1W-EfHbpWSAXVps61Z0xhRlJJjpjSoLBHoZGv0StREAs1024-rj.png?ex=69a86d9e&is=69a71c1e&hm=771595524952303569f8952433d72d3db3ee328c3bf0ad2a04778d1d4eab63fb&=&format=webp&quality=lossless&width=1280&height=859"

DEFAULT_IMAGE_URL = "https://media.istockphoto.com/id/1990444472/photo/scandinavian-style-cozy-living-room-interior.jpg?s=1024x1024&w=is&k=20&c=kHJB-lnK-XmClW7tcWCO68POsyd6H3v0RA5IWCEODb4="

DEFAULT_INSTRUCTION = """Instruction:
Current Odometry:
X: 1.2 meters
Y: 3.4 meters
Absolute Heading: 90.0 degrees

Examine the attached front-camera image. Based on your current position and what you see, output the JSON to execute the next immediate movement.
"""


def download_image(url: str, timeout_s: int = 30) -> bytes:
    # Download image bytes from a URL (supports redirects)
    r = requests.get(url, timeout=timeout_s, allow_redirects=True)
    r.raise_for_status()
    return r.content


def to_png_bytes(image_bytes: bytes) -> bytes:
    # Convert arbitrary image bytes (e.g., WEBP/JPEG/PNG) into PNG bytes
    with Image.open(BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        out = BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()


def post_to_bridge(bridge_url: str, api_key: str, image_b64: str, instruction: str, timeout_s: int = 60) -> dict:
    # Call the Bridge endpoint with image_b64 + instruction
    payload = {"image_b64": image_b64, "instruction": instruction}
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    r = requests.post(bridge_url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Cosmos Reason2 Bridge with an image URL and instruction.")
    parser.add_argument("--bridge-url", default="http://localhost:8080/v1/reason2/action", help="Bridge endpoint URL")
    parser.add_argument("--api-key", default=os.environ.get("BRIDGE_API_KEY", "super-secret-key"), help="API key for Bridge")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="Input image URL")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    args = parser.parse_args()

    try:
        raw_bytes = download_image(args.image_url, timeout_s=args.timeout)
        png_bytes = to_png_bytes(raw_bytes)
        image_b64 = base64.b64encode(png_bytes).decode("ascii")

        result = post_to_bridge(
            bridge_url=args.bridge_url,
            api_key=args.api_key,
            image_b64=image_b64,
            instruction=DEFAULT_INSTRUCTION,
            timeout_s=args.timeout,
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    except requests.HTTPError as e:
        # Print server error body when available
        resp = getattr(e, "response", None)
        if resp is not None:
            print(f"HTTPError: {resp.status_code}\n{resp.text}", file=sys.stderr)
        else:
            print(f"HTTPError: {str(e)}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

