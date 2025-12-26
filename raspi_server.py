import io
import base64
import numpy as np
import cv2
import threading
import time
import os
import logging
import subprocess
import re
import requests
from flask import Flask, request, jsonify, Response, render_template_string, render_template
import config
from app_detect import detect, upload_event_to_cloud
import signal

# --- Flask app ---
app = Flask(__name__)

# --- Logging & directories ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PiCameraServer")
if not os.path.exists(config.SAVE_DIR):
    os.makedirs(config.SAVE_DIR)

CLASS_NAMES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

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

# --- Simple Tracker ---
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

# --- Parking Monitor ---
class ParkingMonitor:
    def __init__(self):
        self.trackers = {"Camera_1": ByteTrackLite(), "Camera_2": ByteTrackLite()}
        self.timers = {}
        self.last_upload_time = {}
        self.zones = {cam: np.array(points) for cam, points in getattr(config, "PARKING_ZONES", {}).items()}

    def process(self, name, res, frame):
        if name not in self.zones: return
        fh, fw = frame.shape[:2]
        cv2.polylines(frame, [self.zones[name]], True, (0, 0, 255), 2)
        pixel_boxes = [[b[0]*fw, b[1]*fh, b[2]*fw, b[3]*fh] for b in res.xyxy]
        tracked = self.trackers[name].update(pixel_boxes, res.conf, res.cls)
        now = time.time()
        for tid, d in tracked.items():
            x1, y1, x2, y2 = map(int, d['box'])
            label = CLASS_NAMES.get(d['cls'], "OBJ")
            center = ((x1+x2)//2, (y1+y2)//2)
            if d['cls'] == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                continue
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                self.timers.pop((name, tid), None)

    def log_violation(self, cam, tid, label, frame):
        import datetime
        now = datetime.datetime.now()
        date_folder = now.strftime("%B %d, %Y (%A)")
        date_dir = os.path.join(config.SAVE_DIR, date_folder)
        os.makedirs(date_dir, exist_ok=True)
        path = os.path.join(date_dir, f"{cam}-{now.strftime('%H_%M_%S')}.jpg")
        cv2.imwrite(path, frame)
        meta = {"tracker_id": tid, "label": label, "timestamp": now.isoformat()}
        upload_event_to_cloud(cam, frame, meta)

# --- Stream handler ---
class Stream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame_buffer = None
        self.last_update = 0
        self.reconnecting = False
        self.read_lock = threading.Lock()
        self.reconnect_event = threading.Event()
        self.running = True
        threading.Thread(target=self._io_thread, daemon=True).start()

    def _io_thread(self):
        while self.running:
            if self.reconnect_event.is_set():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
                self.reconnect_event.clear()
            ret, f = self.cap.read()
            if ret:
                with self.read_lock:
                    self.frame_buffer = f
                    self.last_update = time.time()
                self.reconnecting = False
            else:
                self.reconnecting = True
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)

    def is_online(self):
        return (time.time() - self.last_update) < 3.0

    def get_frame(self):
        with self.read_lock:
            return self.frame_buffer.copy() if self.frame_buffer is not None else None

    def reconnect(self):
        self.reconnect_event.set()

# --- Initialize ---
monitor = ParkingMonitor()
c1, c2 = Stream(config.CAM1_URL), Stream(config.CAM2_URL)
latest_processed = {"Camera_1": None, "Camera_2": None}
proc_lock = threading.Lock()

def processing_worker(cam_name, stream):
    while True:
        frame = stream.get_frame()
        if frame is not None and stream.is_online():
            try:
                res = detect([frame])
                if res:
                    frame_disp = frame.copy()
                    monitor.process(cam_name, res[0], frame_disp)
                    with proc_lock:
                        latest_processed[cam_name] = frame_disp
            except Exception as e:
                logger.error(f"Detection error: {e}")
        time.sleep(0.1)

threading.Thread(target=processing_worker, args=("Camera_1", c1), daemon=True).start()
threading.Thread(target=processing_worker, args=("Camera_2", c2), daemon=True).start()

def gen_single(stream, cam_name):
    FRAME_INTERVAL = 1.0 / 12  # ~12 FPS, adjust as needed
    while True:
        start_time = time.time()
        with proc_lock:
            frame = latest_processed.get(cam_name)
        if frame is None: frame = stream.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, (1280, 720))
        else:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"{cam_name} OFFLINE", (400, 360), 0, 1.5, (0,0,255), 3)
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        elapsed = time.time() - start_time
        sleep_time = max(0, FRAME_INTERVAL - elapsed)
        time.sleep(sleep_time)

# --- Flask routes ---
@app.route('/video_feed_c1')
def video_feed_c1():
    return Response(gen_single(c1, "Camera_1"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_c2')
def video_feed_c2():
    return Response(gen_single(c2, "Camera_2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera_status')
def camera_status():
    return jsonify({
        "Camera_1": {"reconnecting": c1.reconnecting, "online": c1.is_online()},
        "Camera_2": {"reconnecting": c2.reconnecting, "online": c2.is_online()}
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    img = None
    if 'image' in request.files:
        img = decode_image(request.files['image'])
    else:
        data = request.get_json()
        img = decode_image(data.get('image', '')) if data else None

    if img is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        results = detect([img])
        if not results:
            raise RuntimeError("Detection returned no results")
        res = results[0]
        return jsonify({
            'success': True,
            'boxes': res.xyxy.tolist(),
            'confidences': res.conf.tolist(),
            'classes': res.cls.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def decode_image(data):
    if isinstance(data, str):
        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    elif hasattr(data, 'read'):
        img_bytes = data.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return None

@app.route('/')
def index():
    public_url = app.config.get("PUBLIC_URL", "")
    index_html = open("templates/index.html").read()
    # Inject dynamic RASPI_BASE URL
    html_with_url = index_html.replace(
        'const RASPI_BASE = "{{ public_url }}";',
        f'const RASPI_BASE = "{public_url}";'
    )
    return render_template_string(html_with_url)

# --- Graceful shutdown ---
def shutdown(sig, frame):
    print("Shutting down...")
    os._exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on 0.0.0.0:{port}")

    # Start cloudflared automatically
    try:
        cf_proc, public_url = start_cloudflared(port)
        app.config["PUBLIC_URL"] = public_url

        # --- Notify Railway app of the public URL ---
        RAILWAY_API_URL = os.environ.get("RAILWAY_API_URL", "https://web-production-787ca.up.railway.app")
        try:
            resp = requests.post(
                f"{RAILWAY_API_URL}/api/set_pi_url",
                json={"public_url": public_url},
                timeout=5
            )
            print("Posted public URL to Railway:", resp.status_code, resp.text)
        except Exception as e:
            print("Failed to notify Railway app:", e)

    except Exception as e:
        print("Failed to start Cloudflare Tunnel:", e)
        app.config["PUBLIC_URL"] = ""

    app.run(host='0.0.0.0', port=port, threaded=True)
