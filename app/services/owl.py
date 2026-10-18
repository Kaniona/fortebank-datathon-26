"""
Локальный открытый словарь: OWLv2 (Apache-2.0) для проверки конкретных предметов в зоне кассира
(банкноты, планшет, сумка, куртка, чашка и т.д.). Работает полностью локально, веса — в models/owlv2-base.

Применяется точечно: на нескольких кадрах и только внутри кропа рабочей зоны, а не на всём видео.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import MODEL_DIR

logger = logging.getLogger(__name__)

OWL_DIR = Path(MODEL_DIR) / "owlv2-base"


class OwlScorer:
    """Ленивая загрузка; один экземпляр на процесс."""

    def __init__(self, threads: int = 6):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        torch.set_num_threads(threads)
        if not (OWL_DIR / "model.safetensors").exists():
            raise FileNotFoundError(
                f"Нет весов OWLv2 в {OWL_DIR}. Выполните один раз: python scripts/download_models.py"
            )
        self.torch = torch
        # Только локальные веса: во время работы сервис не обращается к сети
        self.processor = Owlv2Processor.from_pretrained(str(OWL_DIR), local_files_only=True)  # nosec B615 — локальный каталог, без сети
        self.model = Owlv2ForObjectDetection.from_pretrained(str(OWL_DIR), local_files_only=True).eval()  # nosec B615
        self._text_cache: dict = {}

    def score(self, crop_bgr: np.ndarray, queries: list[str]) -> dict[str, tuple[float, list[int]]]:
        """
        Для каждого запроса возвращает (максимальная уверенность 0..1, бокс [x1,y1,x2,y2] в пикселях кропа).
        """
        import cv2
        from PIL import Image
        torch = self.torch
        im = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inp = self.processor(text=[queries], images=im, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inp)
        logits = out.logits[0].sigmoid()            # [патчи, запросы]
        boxes = out.pred_boxes[0]                    # cx, cy, w, h (доли квадрата с отступом)
        side = max(im.size)
        res = {}
        for i, q in enumerate(queries):
            j = int(logits[:, i].argmax())
            cx, cy, w, h = [float(v) for v in boxes[j]]
            box = [int((cx - w / 2) * side), int((cy - h / 2) * side), int((cx + w / 2) * side), int((cy + h / 2) * side)]
            res[q] = (float(logits[j, i]), box)
        return res
