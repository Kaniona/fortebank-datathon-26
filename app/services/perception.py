"""
Восприятие кадра: люди (YOLOv8-pose + ByteTrack) и предметы (YOLOv8, COCO).

Все модели выполняются локально на CPU. Весы лежат в каталоге models/;
после их загрузки интернет для обработки видео не требуется.
Запись кадра (dict) одинакова для живого анализа и для кэша tools/extract_features.py.
"""
import logging
from pathlib import Path

import numpy as np

from app.config import MODEL_DIR

logger = logging.getLogger(__name__)

# Предметы COCO, значимые для кассовой зоны
OBJ_CLASSES = {
    67: "cell phone", 73: "book", 41: "cup", 39: "bottle", 24: "backpack", 26: "handbag",
    28: "suitcase", 63: "laptop", 62: "tv", 65: "remote", 46: "banana", 47: "apple",
    49: "orange", 54: "donut", 53: "pizza", 45: "bowl", 40: "wine glass", 75: "vase",
    76: "scissors", 64: "mouse",
}

POSE_WEIGHTS = "yolov8m-pose.pt"
OBJ_WEIGHTS = "yolov8m.pt"
IMGSZ = 960


def _weights(name: str) -> str:
    """Локальные весы из models/, иначе имя для автоскачивания (только при первом запуске)."""
    local = Path(MODEL_DIR) / name
    return str(local) if local.exists() else name


class Perceiver:
    """Обрабатывает кадры по одному; трекинг сохраняется между вызовами."""

    def __init__(self):
        from ultralytics import YOLO
        self.pose = YOLO(_weights(POSE_WEIGHTS))
        self.det = YOLO(_weights(OBJ_WEIGHTS))

    def reset_tracking(self):
        """Сбрасывает состояние трекера: новое видео не должно наследовать ID предыдущего."""
        self.pose.predictor = None

    def process(self, frame: np.ndarray, idx: int, fps: float, with_objects: bool = True) -> dict:
        rp = self.pose.track(frame, imgsz=IMGSZ, conf=0.25, persist=True,
                             tracker="bytetrack.yaml", verbose=False)[0]
        persons = []
        if rp.boxes is not None and len(rp.boxes):
            ids = rp.boxes.id.int().tolist() if rp.boxes.id is not None else [-1] * len(rp.boxes)
            for k, b in enumerate(rp.boxes):
                persons.append({
                    "id": ids[k],
                    "box": [round(v, 1) for v in b.xyxy[0].tolist()],
                    "conf": round(float(b.conf), 3),
                    "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)]
                             for x, y, c in rp.keypoints.data[k].tolist()],
                })
        objs = []
        if with_objects:
            rd = self.det(frame, imgsz=IMGSZ, conf=0.2, classes=list(OBJ_CLASSES), verbose=False)[0]
            objs = [{"cls": OBJ_CLASSES[int(b.cls)], "box": [round(v, 1) for v in b.xyxy[0].tolist()],
                     "conf": round(float(b.conf), 3)} for b in rd.boxes]
        return {"t": round(idx / fps, 2) if fps > 0 else 0.0, "idx": idx, "persons": persons, "objs": objs}
