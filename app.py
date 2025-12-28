import os
import logging
import cv2
import importlib
import traceback
import subprocess
import re
import requests
from flask import Flask, render_template, jsonify, request, make_response, Response, render_template_string, stream_with_context, after_this_request
from datetime import datetime

# NOTE: This app runs on Railway and acts as a relay/config interface.
# All /api/* endpoints are served by the Raspberry Pi server via the cloud link.
# The frontend (index.html) communicates with the Pi via the public URL.

# --- Global Configuration ---
DEFAULT_PORT = int(os.environ.get("PORT", 5000))
DEFAULT_RASPI_IP = os.environ.get("RASPI_IP", "192.168.18.32")
DEFAULT_RASPI_PORT = os.environ.get("RASPI_PORT", "5000")
DEFAULT_RAILWAY_API_URL = os.environ.get("RAILWAY_API_URL", "https://web-production-787ca.up.railway.app")
CLOUDFLARE_TUNNEL_CMD = ["cloudflared", "tunnel", "--url", f"http://localhost:{DEFAULT_PORT}"]
CAMERA_FRAME_SIZE = (640, 360)
BLANK_FRAME_PATH = "blank.jpg"
STATIC_EVENTS_DIR = "static/events"
EVENT_IMAGE_FORMAT = "{camera_id}_{timestamp}.jpg"
EVENT_IMAGE_TIMESTAMP_REPL = lambda ts: ts.replace(':','-').replace('.','-')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")

app = Flask(__name__)

# --- In-memory storage for events ---
EVENTS = []

# --- Load config ---
import config

def get_current_settings():
    return {
        "VIOLATION_TIME_THRESHOLD": getattr(config, "VIOLATION_TIME_THRESHOLD", 10),
        "REPEAT_CAPTURE_INTERVAL": getattr(config, "REPEAT_CAPTURE_INTERVAL", 60),
        "PARKING_ZONES": getattr(config, "PARKING_ZONES", {})
    }

def update_config_py(new_settings):
    import re, json as pyjson
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, "r") as f:
        lines = f.readlines()

    def replace_line(key, value):
        pattern = re.compile(rf"^{key}\s*=\s*.*$")
        for i, line in enumerate(lines):
            if pattern.match(line):
                if key == "PARKING_ZONES":
                    lines[i] = f"{key} = {pyjson.dumps(value)}\n"
                else:
                    lines[i] = f"{key} = {value}\n"
                return
        lines.append(f"{key} = {pyjson.dumps(value) if key=='PARKING_ZONES' else value}\n")

    replace_line("VIOLATION_TIME_THRESHOLD", new_settings.get("VIOLATION_TIME_THRESHOLD", getattr(config, "VIOLATION_TIME_THRESHOLD", 10)))
    replace_line("REPEAT_CAPTURE_INTERVAL", new_settings.get("REPEAT_CAPTURE_INTERVAL", getattr(config, "REPEAT_CAPTURE_INTERVAL", 60)))

    if "PARKING_ZONES" in new_settings:
        current_zones = getattr(config, "PARKING_ZONES", {})
        updated_zones = current_zones.copy()
        for cam, val in new_settings["PARKING_ZONES"].items():
            if val is None:
                updated_zones.pop(cam, None)
            else:
                updated_zones[cam] = val
        replace_line("PARKING_ZONES", updated_zones)

    with open(config_path, "w") as f:
        f.writelines(lines)
    importlib.reload(config)

# --- Camera setup ---
camera_1 = cv2.VideoCapture(0)
camera_2 = cv2.VideoCapture(1)

# Fallback blank frame if camera fails
if not os.path.exists(BLANK_FRAME_PATH):
    import numpy as np
    cv2.imwrite(BLANK_FRAME_PATH, np.zeros((CAMERA_FRAME_SIZE[1], CAMERA_FRAME_SIZE[0], 3), dtype=np.uint8))

def gen(camera):
    blank_frame = cv2.imread(BLANK_FRAME_PATH)
    while True:
        ret, frame = camera.read()
        if not ret:
            frame = blank_frame
        else:
            frame = cv2.resize(frame, CAMERA_FRAME_SIZE)
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# --- Start Cloudflare Tunnel ---
def start_cloudflared(port=DEFAULT_PORT):
    """Start cloudflared tunnel and return public URL."""
    process = subprocess.Popen(
        CLOUDFLARE_TUNNEL_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    url = None
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
        match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            break
    if not url:
        raise RuntimeError("Failed to start cloudflared tunnel")
    print(f"Cloudflared tunnel running at: {url}")
    return process, url

# --- Store latest Pi public URL in memory ---
PI_PUBLIC_URL = ""

@app.route('/api/set_pi_url', methods=['POST'])
def set_pi_url():
    global PI_PUBLIC_URL
    data = request.get_json(force=True)
    PI_PUBLIC_URL = data.get("public_url", "")
    return jsonify({"success": True, "public_url": PI_PUBLIC_URL})

@app.route('/api/get_pi_url')
def get_pi_url():
    # Always return the latest public URL
    return jsonify({"public_url": PI_PUBLIC_URL})

@app.route('/api/pi_public_url')
def pi_public_url():
    logger.info(f"Pi public URL requested: {PI_PUBLIC_URL}")
    return jsonify({"public_url": PI_PUBLIC_URL})

@app.route('/api/cloud_link_status')
def cloud_link_status():
    """Return whether the cloud link (Pi public URL) is set."""
    return jsonify({"cloud_link_active": bool(PI_PUBLIC_URL)})

# --- CORS support ---
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response

@app.after_request
def after_request_func(response):
    return add_cors_headers(response)

# --- Routes ---
@app.route('/')
def index():
    # Always inject the correct RASPI_BASE for the frontend
    return render_template('index.html', public_url=PI_PUBLIC_URL or "")

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route('/violations')
def violations_page():
    return render_template('violations.html')

@app.route('/ping')
def ping():
    return "pong"

@app.route('/api/settings', methods=['GET','POST'])
def api_settings():
    try:
        pi_base = get_pi_base()
        url = f"{pi_base}/api/settings"
        if request.method == 'GET':
            resp = requests.get(url, timeout=10)
            logger.info("Proxy /api/settings GET: Pi returned %s %s", resp.status_code, resp.text)
            # Ensure correct content type and pass-through
            return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
        else:
            resp = requests.post(url, json=request.get_json(force=True), timeout=10)
            logger.info("Proxy /api/settings POST: Pi returned %s %s", resp.status_code, resp.text)
            return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        logger.error(f"Proxy settings error: {e}")
        return jsonify({"success": False, "error": str(e)}), 502

@app.route('/api/raspi_ip')
def raspi_ip():
    return jsonify({"ip": DEFAULT_RASPI_IP, "port": DEFAULT_RASPI_PORT})

@app.route('/api/upload_event', methods=['POST'])
def upload_event():
    try:
        data = request.get_json(force=True)
        camera_id = data.get("camera_id")
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())
        image_b64 = data.get("image")
        meta = data.get("meta", {})

        if not os.path.exists(STATIC_EVENTS_DIR):
            os.makedirs(STATIC_EVENTS_DIR)
        fname = EVENT_IMAGE_FORMAT.format(
            camera_id=camera_id,
            timestamp=EVENT_IMAGE_TIMESTAMP_REPL(timestamp)
        )
        img_path = os.path.join(STATIC_EVENTS_DIR, fname)

        import base64
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        EVENTS.append({
            "camera_id": camera_id,
            "timestamp": timestamp,
            "image_url": f"/{STATIC_EVENTS_DIR}/{fname}",
            "meta": meta
        })
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Upload event failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/events')
def api_events():
    return jsonify(EVENTS)

def get_pi_base():
    # Returns the current Pi public URL, or raises if not set
    if not PI_PUBLIC_URL:
        raise RuntimeError("Pi public URL not set")
    return PI_PUBLIC_URL.rstrip('/')

@app.route('/api/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def proxy_api(path):
    if request.method == 'OPTIONS':
        return add_cors_headers(jsonify({})), 200
    try:
        pi_base = get_pi_base()
        url = f"{pi_base}/api/{path}"
        if request.method == 'GET':
            resp = requests.get(url, params=request.args, timeout=10)
        else:
            resp = requests.post(url, json=request.get_json(force=True), timeout=10)
        proxy_response = Response(resp.content, resp.status_code, resp.headers.items())
        return add_cors_headers(proxy_response)
    except Exception as e:
        logger.error(f"Proxy API error: {e}")
        return add_cors_headers(jsonify({"success": False, "error": str(e)})), 502

@app.route('/video_feed_c1')
def proxy_video_feed_c1():
    try:
        pi_base = get_pi_base()
        url = f"{pi_base}/video_feed_c1"
        resp = requests.get(url, stream=True, timeout=10)
        proxy_response = Response(stream_with_context(resp.iter_content(chunk_size=4096)),
                                  content_type=resp.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame'))
        return add_cors_headers(proxy_response)
    except Exception as e:
        logger.error(f"Proxy video_feed_c1 error: {e}")
        return add_cors_headers(Response("Camera feed unavailable", 502))

@app.route('/video_feed_c2')
def proxy_video_feed_c2():
    try:
        pi_base = get_pi_base()
        url = f"{pi_base}/video_feed_c2"
        resp = requests.get(url, stream=True, timeout=10)
        proxy_response = Response(stream_with_context(resp.iter_content(chunk_size=4096)),
                                  content_type=resp.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame'))
        return add_cors_headers(proxy_response)
    except Exception as e:
        logger.error(f"Proxy video_feed_c2 error: {e}")
        return add_cors_headers(Response("Camera feed unavailable", 502))

@app.route('/api/camera_status', methods=['GET', 'OPTIONS'])
def api_camera_status():
    if request.method == 'OPTIONS':
        return add_cors_headers(jsonify({})), 200
    try:
        pi_base = get_pi_base()
        url = f"{pi_base}/api/camera_status"
        resp = requests.get(url, timeout=10)
        try:
            data = resp.json()
        except Exception:
            logger.error(f"Invalid JSON from Pi camera_status: {resp.text[:200]}")
            return add_cors_headers(jsonify({"success": False, "error": "Invalid response from Pi server"})), 502
        return add_cors_headers(jsonify(data))
    except Exception as e:
        logger.error(f"Proxy camera_status error: {e}")
        return add_cors_headers(jsonify({"success": False, "error": str(e)})), 502

# --- Error handler ---
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error("Unhandled Exception: %s\n%s", e, traceback.format_exc())
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": str(e)}), 500
    return make_response("Internal Server Error", 500)

# --- Main ---
if __name__=="__main__": 
    port = DEFAULT_PORT

    # Start cloudflared and get public URL
    try:
        cf_proc, public_url = start_cloudflared(port)
        app.config["PUBLIC_URL"] = public_url
    except Exception as e:
        print("Failed to start Cloudflare Tunnel:", e)
        app.config["PUBLIC_URL"] = ""

    app.run(host='0.0.0.0', port=port, threaded=True)
