import cv2
import threading
import time
import os
import numpy as np
import logging
from flask import Flask, Response, render_template, jsonify, request
import json
import config
import datetime
import queue
import subprocess
import importlib

# --- Setup Logging & Folders ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParkingApp")
if not os.path.exists(config.SAVE_DIR):
    os.makedirs(config.SAVE_DIR)

CLASS_NAMES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# Settings management (now only via config.py)
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
        # If not found, append
        if key == "PARKING_ZONES":
            lines.append(f"{key} = {pyjson.dumps(value)}\n")
        else:
            lines.append(f"{key} = {value}\n")
    replace_line("VIOLATION_TIME_THRESHOLD", new_settings.get("VIOLATION_TIME_THRESHOLD", getattr(config, "VIOLATION_TIME_THRESHOLD", 10)))
    replace_line("REPEAT_CAPTURE_INTERVAL", new_settings.get("REPEAT_CAPTURE_INTERVAL", getattr(config, "REPEAT_CAPTURE_INTERVAL", 60)))
    # Only update PARKING_ZONES if present in new_settings
    if "PARKING_ZONES" in new_settings:
        current_zones = getattr(config, "PARKING_ZONES", {})
        updated_zones = current_zones.copy()
        for cam, val in new_settings["PARKING_ZONES"].items():
            if val is None:
                # Remove the camera zone
                updated_zones.pop(cam, None)
            else:
                updated_zones[cam] = val
        replace_line("PARKING_ZONES", updated_zones)
    with open(config_path, "w") as f:
        f.writelines(lines)
    # Reload config module
    importlib.reload(config)

class ByteTrackLite:
    def __init__(self):
        self.tracked_objects = {}
        self.frame_count = 0
        self.next_id = 0
        self.buffer = 30

    def get_iou(self, b1, b2):
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, boxes, scores, clss):
        self.frame_count += 1
        new_tracks = {}
        for box, score, cid in zip(boxes, scores, clss):
            best_id, best_iou = None, 0.3
            for tid, t in self.tracked_objects.items():
                iou = self.get_iou(box, t['box'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None:
                new_tracks[best_id] = {'box': box, 'cls': cid, 'last_seen': self.frame_count}
                self.tracked_objects.pop(best_id, None)
            elif score >= config.DETECTION_THRESHOLD:
                new_tracks[self.next_id] = {'box': box, 'cls': cid, 'last_seen': self.frame_count}
                self.next_id += 1

        for tid, t in self.tracked_objects.items():
            if self.frame_count - t['last_seen'] < self.buffer:
                new_tracks[tid] = t

        self.tracked_objects = new_tracks
        return {k: v for k, v in new_tracks.items() if v['last_seen'] == self.frame_count}

class ParkingMonitor:
    def __init__(self):
        self.trackers = {"Camera_1": ByteTrackLite(), "Camera_2": ByteTrackLite()}
        self.timers = {}
        self.last_upload_time = {}
        # self.traces = {"Camera_1": {}, "Camera_2": {}}  # Removed trace storage
        # self.trace_length = 30  # Removed trace length
        self.reload_zones()

    def reload_zones(self):
        importlib.reload(config)
        self.zones = {
            cam: np.array(points)
            for cam, points in getattr(config, "PARKING_ZONES", {}).items()
        }

    def process(self, name, res, frame):
        # Always reload zones before processing to reflect latest config.py changes
        self.reload_zones()
        fh, fw = frame.shape[:2]
        # Change zone color to red
        cv2.polylines(frame, [self.zones[name]], True, (0, 0, 255), 2)
        
        pixel_boxes = [[b[0]*fw, b[1]*fh, b[2]*fw, b[3]*fh] for b in res.xyxy]
        tracked = self.trackers[name].update(pixel_boxes, res.conf, res.cls)
        now = time.time()

        # --- Trace logic removed ---

        for tid, d in tracked.items():
            x1, y1, x2, y2 = map(int, d['box'])
            label = CLASS_NAMES.get(d['cls'], "OBJ")
            center = ((x1+x2)//2, (y1+y2)//2)

            # Person detection (Yellow box, no timer)
            if d['cls'] == 0:
                # Optimize: use thinner rectangle and skip putText for less lag
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                # Comment out or remove the following line to reduce lag:
                # cv2.putText(frame, f"{label} #{tid}", (x1, y1-8), 0, 0.6, (255, 255, 0), 2)
                continue

            # Vehicle detection: always draw, but only time/countdown if in zone
            in_zone = cv2.pointPolygonTest(self.zones[name], center, False) >= 0
            if in_zone:
                self.timers.setdefault((name, tid), now)
                dur = int(now - self.timers[(name, tid)])
                is_violation = dur >= config.VIOLATION_TIME_THRESHOLD
                color = (0, 0, 255) if is_violation else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} #{tid}: {dur}s", (x1, y1-8), 0, 0.6, color, 2)
                if is_violation:
                    last_up = self.last_upload_time.get((name, tid), 0)
                    if now - last_up > config.REPEAT_CAPTURE_INTERVAL:
                        self.log_violation(name, tid, label, frame)
                        self.last_upload_time[(name, tid)] = now
            else:
                # Draw detected vehicle outside zone (blue box, no timer/countdown)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} #{tid}", (x1, y1-8), 0, 0.6, (255, 0, 0), 2)
                self.timers.pop((name, tid), None)

    def log_violation(self, cam, tid, label, frame):
        ts = int(time.time())
        now = datetime.datetime.now()
        date_folder = now.strftime("%B %d, %Y (%A)")
        date_dir = os.path.join(config.SAVE_DIR, date_folder)
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        # Format time as HH_MM_ss for filename safety
        time_str = now.strftime("%H_%M_%S")
        filename = f"{cam}-{time_str}.jpg"
        path = os.path.join(date_dir, filename)
        cv2.imwrite(path, frame)
        logger.info(f"Violation Logged: {label} on {cam} (saved to {path})")

class Stream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.last_update = None  # Track last frame update time
        self.reconnect_event = threading.Event()
        self.reconnecting = False  # Track reconnecting state
        self.read_lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        # Try to read frames as fast as possible for higher FPS
        while self.running:
            if self.reconnect_event.is_set():
                self.reconnecting = True
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
                self.reconnect_event.clear()
            ret, f = self.cap.read()
            if ret:
                with self.read_lock:
                    self.frame = f
                    self.last_update = time.time()
                self.reconnecting = False
            else:
                self.reconnecting = True
                time.sleep(0.2)  # Shorter sleep for faster reconnect attempts
                self.cap = cv2.VideoCapture(self.url)

    def is_online(self, timeout=2.0):
        """Returns True if the stream has updated recently."""
        with self.read_lock:
            return self.last_update is not None and (time.time() - self.last_update) < timeout

    def reconnect(self):
        self.reconnect_event.set()

    def get_frame(self):
        with self.read_lock:
            return None if self.frame is None else self.frame.copy()

app = Flask(__name__)
monitor = ParkingMonitor()
c1, c2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)

# Shared latest processed frames for each camera
latest_frames = {"Camera_1": None, "Camera_2": None}
latest_frames_lock = threading.Lock()
latest_detection_results = {"Camera_1": None, "Camera_2": None}
latest_detection_lock = threading.Lock()

DETECTION_INTERVAL = 0.1  # seconds between detections (increase for less CPU, decrease for more FPS)

# Dynamically import the correct detect function based on config.HAILO
if getattr(config, "HAILO", False):
    from app_detect import detect
else:
    try:
        from ultralytics import YOLO
        class DetectionResult:
            def __init__(self, xyxy, confs, clss):
                self.xyxy = xyxy
                self.conf = confs
                self.cls = clss
        class YOLODetector:
            def __init__(self, model_path):
                self.model = YOLO(model_path)
                self.monitored_classes = [0, 2, 3, 5, 7]
            def postprocess(self, results, frame_shape):
                all_boxes, all_confs, all_clss = [], [], []
                h, w = frame_shape[:2]
                for r in results:
                    for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                        cls = int(cls)
                        if cls not in self.monitored_classes:
                            continue
                        if conf < config.DETECTION_THRESHOLD:
                            continue
                        xmin, ymin, xmax, ymax = box
                        all_boxes.append([xmin/w, ymin/h, xmax/w, ymax/h])
                        all_confs.append(conf)
                        all_clss.append(cls)
                return DetectionResult(np.array(all_boxes), np.array(all_confs), np.array(all_clss))
            def run_detection(self, frames):
                results = []
                for frame in frames:
                    yolo_results = self.model.predict(frame, verbose=False)
                    results.append(self.postprocess(yolo_results, frame.shape))
                return results
        _detector = None
        def detect(frames):
            global _detector
            if _detector is None:
                _detector = YOLODetector(config.MODEL_PATH)
            return _detector.run_detection(frames)
    except ImportError:
        # Fallback: detection is disabled, log warning
        logging.warning("ultralytics not installed. Detection is disabled.")
        class DetectionResult:
            def __init__(self, xyxy=None, confs=None, clss=None):
                self.xyxy = np.array([]) if xyxy is None else xyxy
                self.conf = np.array([]) if confs is None else confs
                self.cls = np.array([]) if clss is None else clss
        def detect(frames):
            # Return empty detection results for each frame
            return [DetectionResult() for _ in frames]

def detection_worker(cam_name, stream):
    while True:
        frame = stream.get_frame()
        if frame is not None and stream.is_online():
            results = detect([frame])
            with latest_detection_lock:
                latest_detection_results[cam_name] = (results[0], frame)
        time.sleep(DETECTION_INTERVAL)

def overlay_worker(cam_name):
    while True:
        with latest_detection_lock:
            detection = latest_detection_results.get(cam_name)
        if detection is not None:
            res, frame = detection
            frame_disp = frame.copy()
            monitor.process(cam_name, res, frame_disp)
            with latest_frames_lock:
                latest_frames[cam_name] = cv2.resize(frame_disp, (640, 480))
        else:
            with latest_frames_lock:
                latest_frames[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)
        time.sleep(0.01)

# Start detection and overlay threads for each camera
threading.Thread(target=detection_worker, args=("Camera_1", c1), daemon=True).start()
threading.Thread(target=detection_worker, args=("Camera_2", c2), daemon=True).start()
threading.Thread(target=overlay_worker, args=("Camera_1",), daemon=True).start()
threading.Thread(target=overlay_worker, args=("Camera_2",), daemon=True).start()

def gen():
    while True:
        with latest_frames_lock:
            frame1 = latest_frames.get("Camera_1")
            frame2 = latest_frames.get("Camera_2")
            out = []
            if frame1 is not None:
                out.append(frame1)
            if frame2 is not None:
                out.append(frame2)
            if not out:
                # fallback placeholder
                offline_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(offline_placeholder, "CAMERA OFFLINE", (60, 240), 0, 1.2, (0,0,255), 3, cv2.LINE_AA)
                out = [offline_placeholder, offline_placeholder]
            elif len(out) == 1:
                offline_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(offline_placeholder, "CAMERA OFFLINE", (60, 240), 0, 1.2, (0,0,255), 3, cv2.LINE_AA)
                out.append(offline_placeholder)
            combined = cv2.hconcat(out)
            _, buf = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

def gen_single(cam, cam_name):
    offline_placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(offline_placeholder, f"{cam_name} OFFLINE", (120, 360), 0, 2.2, (0,0,255), 5, cv2.LINE_AA)
    while True:
        with latest_frames_lock:
            frame = latest_frames.get(cam_name)
            if frame is not None:
                out = cv2.resize(frame, (1280, 720))
            else:
                out = offline_placeholder
            _, buf = cv2.imencode('.jpg', out)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    # (Optional: keep for backward compatibility, or remove if not needed)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c1')
def video_feed_c1():
    return Response(gen_single(c1, "Camera_1"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c2')
def video_feed_c2():
    return Response(gen_single(c2, "Camera_2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/violations.html')
def violations_page():
    return render_template('violations.html')

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(get_current_settings())

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    update_config_py(data)
    return jsonify({"success": True})

@app.route('/api/reconnect/<camera>', methods=['POST'])
def reconnect_camera(camera):
    if camera == "Camera_1":
        c1.reconnect()
        return jsonify({"success": True, "message": "Camera_1 reconnect triggered"})
    elif camera == "Camera_2":
        c2.reconnect()
        return jsonify({"success": True, "message": "Camera_2 reconnect triggered"})
    else:
        return jsonify({"success": False, "message": "Unknown camera"}), 400

@app.route('/api/camera_status')
def camera_status():
    return jsonify({
        "Camera_1": {
            "reconnecting": bool(getattr(c1, "reconnecting", False))
        },
        "Camera_2": {
            "reconnecting": bool(getattr(c2, "reconnecting", False))
        }
    })

@app.route('/api/zone_selector', methods=['POST'])
def api_zone_selector():
    try:
        data = request.get_json(force=True)
        camera = data.get("camera", "Camera_1")
        cam_arg = "1" if camera == "Camera_1" else "2"
        # Run zone_selector.py and capture output
        proc = subprocess.Popen(
            ["python", "zone_selector.py", cam_arg],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate()
        # Parse output for zone coordinates (expects a line with JSON array)
        import re, json as pyjson
        match = re.search(r"\[\s*\[.*?\]\s*\]", stdout, re.DOTALL)
        if match:
            zone = pyjson.loads(match.group(0))
            # Update config and return new zone
            update_config_py({"PARKING_ZONES": {camera: zone}})
            return jsonify({"success": True, "zone": zone})
        else:
            return jsonify({"success": False, "error": "No zone found", "stdout": stdout, "stderr": stderr})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)