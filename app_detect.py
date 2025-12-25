import numpy as np
import cv2
import threading
import logging
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
import config

logger = logging.getLogger("ParkingApp")

class DetectionResult:
    def __init__(self, xyxy, confs, clss):
        self.xyxy = xyxy      
        self.conf = confs     
        self.cls = clss       

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

        # COCO IDs for Person, Car, Motorcycle, Bus, Truck
        self.monitored_classes = [0, 2, 3, 5, 7]

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def postprocess(self, raw_out):
        all_boxes, all_confs, all_clss = [], [], []
        nms_keys = [k for k in raw_out.keys() if 'nms' in k.lower() or 'output' in k.lower()]
        
        if not nms_keys:
            return DetectionResult(np.array([]), np.array([]), np.array([]))

        # Access detections (usually a list containing one array per batch)
        detections_by_class = raw_out[nms_keys[0]]
        batch_detections = detections_by_class[0] 

        # Iterate through the 80 classes provided by Hailo NMS
        for class_id, class_detections in enumerate(batch_detections):
            if class_id not in self.monitored_classes:
                continue
                
            for det in class_detections:
                score = float(det[4])
                if score >= config.DETECTION_THRESHOLD:
                    # Hailo format: [ymin, xmin, ymax, xmax]
                    # We store as [xmin, ymin, xmax, ymax] for our app logic
                    all_boxes.append([det[1], det[0], det[3], det[2]])
                    all_confs.append(score)
                    all_clss.append(class_id)

        # Now, all monitored classes are detected regardless of zone.
        # No filtering by illegal parking zone here; that logic should be handled elsewhere.

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
