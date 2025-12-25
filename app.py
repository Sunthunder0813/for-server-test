import os
import logging
from flask import Flask, render_template, jsonify, request, make_response
import config
import importlib
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")

app = Flask(__name__)

def get_current_settings():
    return {
        "VIOLATION_TIME_THRESHOLD": getattr(config, "VIOLATION_TIME_THRESHOLD", 10),
        "REPEAT_CAPTURE_INTERVAL": getattr(config, "REPEAT_CAPTURE_INTERVAL", 60),
        "PARKING_ZONES": getattr(config, "PARKING_ZONES", {})
    }

def update_config_py(new_settings):
    import re
    import json as pyjson
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
        lines.append(f"{key} = {pyjson.dumps(value) if key == 'PARKING_ZONES' else value}\n")
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

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/violations.html')
def violations_page():
    return render_template('violations.html')

@app.route('/ping')
def ping():
    return "pong"

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'POST':
        update_config_py(request.get_json())
        return jsonify({"success": True})
    return jsonify(get_current_settings())

@app.route('/api/zone_selector', methods=['POST'])
def api_zone_selector():
    try:
        data = request.get_json(force=True)
        camera = data.get("camera", "Camera_1")
        cam_arg = "1" if camera == "Camera_1" else "2"
        import subprocess
        proc = subprocess.Popen(["python", "zone_selector.py", cam_arg], cwd=os.path.dirname(__file__),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        import re, json as pyjson
        match = re.search(r"\[\s*\[.*?\]\s*\]", stdout, re.DOTALL)
        if match:
            zone = pyjson.loads(match.group(0))
            update_config_py({"PARKING_ZONES": {camera: zone}})
            return jsonify({"success": True, "zone": zone})
        return jsonify({"success": False, "error": "No zone found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/raspi_ip')
def raspi_ip():
    # For ngrok, only the hostname is needed; frontend will use HTTPS and omit port.
    raspi_ip = os.environ.get("RASPI_IP", "192.168.18.32")
    # raspi_port is kept for compatibility, but frontend should ignore it for ngrok
    raspi_port = os.environ.get("RASPI_PORT", "5000")
    return jsonify({"ip": raspi_ip, "port": raspi_port})

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the full traceback
    logger.error("Unhandled Exception: %s\n%s", e, traceback.format_exc())
    # For API endpoints, return JSON error
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": str(e)}), 500
    # For web pages, show a simple error page
    return make_response("Internal Server Error", 500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
