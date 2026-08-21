"""
Real-time person detection and tracking using YOLOv8n + ByteTrack.
Each person gets a persistent track_id while visible, which is essential for
determining their movement direction (in/out) and for collecting the best face images
as they pass through the frame.
"""
import logging
from ultralytics import YOLO
from src.device_utils import resolve_device

logger = logging.getLogger("attendance.detector")

PERSON_CLASS_ID = 0 


class DetectorTracker:
    def __init__(self, model_path="models/yolov8n.pt", confidence_threshold=0.5,
                 device="auto", tracker_type="bytetrack"):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = resolve_device(device)
        self.tracker_cfg = "bytetrack.yaml" if tracker_type == "bytetrack" else "botsort.yaml"
        logger.info(f"Loaded detection model: {model_path} on device: {self.device}")

    def track(self, frame):
        """
        Returns a list of all detected persons in the frame:
        [{"track_id": int, "bbox": (x1, y1, x2, y2), "confidence": float}, ...]
        """
        results = self.model.track(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            device=self.device,
            tracker=self.tracker_cfg,
            persist=True,    
            verbose=False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.id is None:
           
            return detections

        for box, track_id, conf in zip(boxes.xyxy.cpu().numpy(),
                                        boxes.id.cpu().numpy(),
                                        boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            detections.append({
                "track_id": int(track_id),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(conf),
            })
        return detections
