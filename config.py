# Camera RTSP URLs
CAM1_URL = "rtsp://192.168.18.2:554/stream"
CAM2_URL = "rtsp://192.168.18.199:554/stream"

# Path to YOLO model
MODEL_PATH = "models/yolov8s.hef"

# Directory to save violation images
SAVE_DIR = "static/violations"

DETECTION_THRESHOLD = 0.3      # Minimum confidence (0.0 to 1.0)
VIOLATION_TIME_THRESHOLD = 10
REPEAT_CAPTURE_INTERVAL = 60
# Define parking zones for each camera

PARKING_ZONES = {
"Camera_1": [[249, 242], [255, 404], [654, 426], [443, 261]],
"Camera_2": [[46, 437], [453, 253], [664, 259], [678, 438]]
}
