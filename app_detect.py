import numpy as np
import cv2
import threading
import logging
import os
import requests
import base64
from datetime import datetime
import config

logger = logging.getLogger("ParkingApp")

# Try to import Hailo SDK
try:
    from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
    HAILO_AVAILABLE = True
except ImportError:
    logger.warning("Hailo SDK not found, falling back to remote or CPU detection.")
    HAILO_AVAILABLE = False

USE_REMOTE_DETECTION = os.environ.get("USE_REMOTE_DETECTION", "0") == "1"
RASPI_URL = os.environ.get("RASPI_URL", "http://192.168.18.32:5000/detect")
CLOUD_URL = os.environ.get("CLOUD_URL", "https://illegal-parking-flask.onrender.com/api/upload_event")

class DetectionResult:
    def __init__(self, xyxy, confs, clss):
        self.xyxy = xyxy
        self.conf = confs
        self.cls = clss

if HAILO_AVAILABLE and not USE_REMOTE_DETECTION:
    # Hailo-based detection
    class HailoDetector:
        def __init__(self, hef_path):
            self.hef = HEF(hef_path)
            self.target = VDevice()
            self.lock = threading.Lock()
            params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.target.configure(self.hef, params)[0]
            self.input_vstreams_params = InputVStreamParams.make(self.network_group)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
            self.input_info = self.hef.get_input_vstream_infos()[0]
            self.height, self.width, _ = self.input_info.shape
            self.monitored_classes = [0, 2, 3, 5, 7]

        def preprocess(self, frame):
            resized = cv2.resize(frame, (self.width, self.height))
            return np.expand_dims(resized, axis=0).astype(np.uint8)

        def postprocess(self, raw_out):
            all_boxes, all_confs, all_clss = [], [], []
            nms_keys = [k for k in raw_out.keys() if 'nms' in k.lower() or 'output' in k.lower()]
            if not nms_keys:
                return DetectionResult(np.array([]), np.array([]), np.array([]))
            detections_by_class = raw_out[nms_keys[0]]
            batch_detections = detections_by_class[0]
            for class_id, class_detections in enumerate(batch_detections):
                if class_id not in self.monitored_classes:
                    continue
                for det in class_detections:
                    score = float(det[4])
                    if score >= config.DETECTION_THRESHOLD:
                        all_boxes.append([det[1], det[0], det[3], det[2]])
                        all_confs.append(score)
                        all_clss.append(class_id)
            return DetectionResult(np.array(all_boxes), np.array(all_confs), np.array(all_clss))

        def run_detection(self, frames):
            results = []
            with self.lock:
                with self.network_group.activate():
                    with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as pipeline:
                        for frame in frames:
                            input_data = {self.input_info.name: self.preprocess(frame)}
                            raw_out = pipeline.infer(input_data)
                            results.append(self.postprocess(raw_out))
            return results

    _detector = None
    def detect(frames):
        global _detector
        if _detector is None:
            _detector = HailoDetector(config.MODEL_PATH)
        return _detector.run_detection(frames)

else:
    # CPU fallback using YOLOv8
    import warnings
    warnings.warn("Hailo detection unavailable, using YOLOv8 CPU fallback.")
    from ultralytics import YOLO

    _model = YOLO(config.MODEL_PATH)  # Make sure MODEL_PATH points to .pt model (not .hef)
    _lock = threading.Lock()

    def detect(frames):
        results = []
        with _lock:
            for frame in frames:
                try:
                    res = _model(frame)[0]
                    if hasattr(res, 'boxes') and len(res.boxes):
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        conf = res.boxes.conf.cpu().numpy()
                        clss = res.boxes.cls.cpu().numpy().astype(int)
                    else:
                        xyxy, conf, clss = np.array([]), np.array([]), np.array([])
                    results.append(DetectionResult(xyxy, conf, clss))
                except Exception as e:
                    logger.error(f"YOLOv8 CPU detection failed: {e}")
                    results.append(DetectionResult(np.array([]), np.array([]), np.array([])))
        return results

def upload_event_to_cloud(camera_id, frame, meta=None):
    """Uploads a violation event to the cloud."""
    try:
        _, buf = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buf).decode('utf-8')
        payload = {
            "camera_id": camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "image": img_b64,
            "meta": meta or {}
        }
        resp = requests.post(CLOUD_URL, json=payload, timeout=10)
        if resp.ok:
            return True
        else:
            logger.error(f"Cloud upload failed: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Exception in upload_event_to_cloud: {e}")
        return False
