import os
import logging
import cv2
import importlib
import traceback
import subprocess
import re
from flask import Flask, render_template, jsonify, request, make_response, Response, render_template_string, stream_with_context
from datetime import datetime
import time

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
import config
CAMERA_1_SRC = os.environ.get("CAMERA_1_SRC", config.CAM1_URL)
CAMERA_2_SRC = os.environ.get("CAMERA_2_SRC", config.CAM2_URL)

camera_1 = cv2.VideoCapture(CAMERA_1_SRC)
camera_2 = cv2.VideoCapture(CAMERA_2_SRC)

# --- Camera status tracking ---
CAMERA_STATUS = {
    "Camera_1": {"last_frame_time": 0, "online": False},
    "Camera_2": {"last_frame_time": 0, "online": False}
}
CAMERA_TIMEOUT = 5  # seconds

# Fallback blank frame if camera fails
blank_frame_path = "blank.png"
if not os.path.exists(blank_frame_path):
    import numpy as np
    cv2.imwrite(blank_frame_path, np.zeros((360,640,3), dtype=np.uint8))

def gen(camera, cam_name):
    blank_frame = cv2.imread(blank_frame_path)
    while True:
        if not camera.isOpened():
            frame = blank_frame.copy()
            cv2.putText(frame, f"{cam_name} OFFLINE", (160, 180), 0, 1.5, (0,0,255), 3)
            CAMERA_STATUS[cam_name]["online"] = False
        else:
            ret, frame = camera.read()
            if not ret:
                frame = blank_frame.copy()
                cv2.putText(frame, f"{cam_name} OFFLINE", (160, 180), 0, 1.5, (0,0,255), 3)
                CAMERA_STATUS[cam_name]["online"] = False
            else:
                frame = cv2.resize(frame, (640,360))
                CAMERA_STATUS[cam_name]["last_frame_time"] = time.time()
                CAMERA_STATUS[cam_name]["online"] = True
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
    return jsonify({"public_url": PI_PUBLIC_URL})

@app.route('/api/pi_public_url')
def pi_public_url():
    logger.info(f"Pi public URL requested: {PI_PUBLIC_URL}")
    return jsonify({"public_url": PI_PUBLIC_URL})

# --- Routes ---
@app.route('/')
def index():
    # Inject the latest Pi public URL if available
    return render_template('index.html', public_url=PI_PUBLIC_URL)

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
    if not camera_1.isOpened():
        return Response("Camera 1 not available", status=503, mimetype='text/plain')
    logger.info(f'{request.remote_addr} - - [{datetime.datetime.now().strftime("%d/%b/%Y %H:%M:%S")}] "GET /video_feed_c1{request.query_string.decode() and "?" + request.query_string.decode() or ""} HTTP/1.1" 200 -')
    return Response(
        stream_with_context(gen(camera_1, "Camera_1")),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={"Cache-Control": "no-store"}
    )

@app.route('/video_feed_c2')
def video_feed_c2():
    if not camera_2.isOpened():
        return Response("Camera 2 not available", status=503, mimetype='text/plain')
    logger.info(f'{request.remote_addr} - - [{datetime.datetime.now().strftime("%d/%b/%Y %H:%M:%S")}] "GET /video_feed_c2{request.query_string.decode() and "?" + request.query_string.decode() or ""} HTTP/1.1" 200 -')
    return Response(
        stream_with_context(gen(camera_2, "Camera_2")),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={"Cache-Control": "no-store"}
    )

@app.route('/api/camera_status')
def camera_status():
    try:
        now = time.time()
        status = {}
        for cam, cam_obj in [("Camera_1", camera_1), ("Camera_2", camera_2)]:
            # If last frame was sent recently, consider online
            online = CAMERA_STATUS[cam]["online"] and (now - CAMERA_STATUS[cam]["last_frame_time"] < CAMERA_TIMEOUT)
            status[cam] = {
                "reconnecting": not online,
                "online": online
            }
        return jsonify(status)
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
