import os
import logging
import cv2
import importlib
import traceback
import subprocess
import re
from flask import Flask, render_template, jsonify, request, make_response, Response, render_template_string
from datetime import datetime

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
blank_frame_path = "blank.jpg"
if not os.path.exists(blank_frame_path):
    import numpy as np
    cv2.imwrite(blank_frame_path, np.zeros((360,640,3), dtype=np.uint8))

def gen(camera):
    blank_frame = cv2.imread(blank_frame_path)
    while True:
        ret, frame = camera.read()
        if not ret:
            frame = blank_frame
        else:
            frame = cv2.resize(frame, (640,360))
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# --- Start Cloudflare Tunnel ---
def start_cloudflared(port=5000):
    """Start cloudflared tunnel and return public URL."""
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
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

# --- Routes ---
@app.route('/')
def index():
    public_url = app.config.get("PUBLIC_URL", "")
    index_html = open("templates/index.html").read()
    # Inject dynamic RASPI_BASE URL
    html_with_url = index_html.replace(
        'const RASPI_BASE = "https://educated-contest-certain-eau.trycloudflare.com";',
        f'const RASPI_BASE = "{public_url}";'
    )
    return render_template_string(html_with_url)

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/violations.html')
def violations_page():
    return render_template('violations.html')

@app.route('/ping')
def ping():
    return "pong"

@app.route('/video_feed_c1')
def video_feed_c1():
    return Response(gen(camera_1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c2')
def video_feed_c2():
    return Response(gen(camera_2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera_status')
def camera_status():
    try:
        return jsonify({
            "Camera_1": {"reconnecting": not camera_1.isOpened()},
            "Camera_2": {"reconnecting": not camera_2.isOpened()}
        })
    except Exception as e:
        logger.error(f"Error checking camera status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/settings', methods=['GET','POST'])
def api_settings():
    if request.method=='POST':
        update_config_py(request.get_json())
        return jsonify({"success": True})
    return jsonify(get_current_settings())

@app.route('/api/raspi_ip')
def raspi_ip():
    raspi_ip = os.environ.get("RASPI_IP","192.168.18.32")
    raspi_port = os.environ.get("RASPI_PORT","5000")
    return jsonify({"ip": raspi_ip,"port":raspi_port})

@app.route('/api/upload_event', methods=['POST'])
def upload_event():
    try:
        data = request.get_json(force=True)
        camera_id = data.get("camera_id")
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())
        image_b64 = data.get("image")
        meta = data.get("meta", {})

        if not os.path.exists("static/events"):
            os.makedirs("static/events")
        fname = f"{camera_id}_{timestamp.replace(':','-').replace('.','-')}.jpg"
        img_path = os.path.join("static/events", fname)

        import base64
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        EVENTS.append({
            "camera_id": camera_id,
            "timestamp": timestamp,
            "image_url": f"/static/events/{fname}",
            "meta": meta
        })
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Upload event failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/events')
def api_events():
    return jsonify(EVENTS)

# --- Error handler ---
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error("Unhandled Exception: %s\n%s", e, traceback.format_exc())
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": str(e)}), 500
    return make_response("Internal Server Error", 500)

# --- Main ---
if __name__=="__main__":
    port = int(os.environ.get("PORT",5000))

    # Start cloudflared and get public URL
    try:
        cf_proc, public_url = start_cloudflared(port)
        app.config["PUBLIC_URL"] = public_url
    except Exception as e:
        print("Failed to start Cloudflare Tunnel:", e)
        app.config["PUBLIC_URL"] = ""

    app.run(host='0.0.0.0', port=port, threaded=True)
