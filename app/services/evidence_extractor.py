"""
Извлечение доказательных кадров из видео.
Сохраняет JPEG-кадры и возвращает их в Base64.
"""
import base64
import logging
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class EvidenceExtractor:
    """
    Извлекает кадры-доказательства из видео.

    Для каждого события извлекается 1–3 кадра:
    - Кадр в момент события
    - Кадр за ~1 секунду до (если возможно)
    - Кадр через ~1 секунду после (если возможно)
    """

    def __init__(self, video_reader):
        self.video_reader = video_reader

    def extract(
        self,
        timestamp_sec: float,
        fps: float,
        frame_indices: Optional[list[int]] = None,
        max_frames: int = 3,
    ) -> list[dict]:
        """
        Извлекает доказательные кадры для события.

        Args:
            timestamp_sec: время события в секундах
            fps: частота кадров
            frame_indices: конкретные индексы кадров (если заданы)
            max_frames: максимум кадров (1–3)

        Returns:
            список словарей: {
                "timestamp_sec": float,
                "image_base64": str (JPEG в Base64)
            }
        """
        results: list[dict] = []

        if frame_indices:
            indices = frame_indices[:max_frames]
        else:
            # Центральный кадр + соседи
            center_idx = int(timestamp_sec * fps)
            offset = max(1, int(fps))
            indices = [center_idx]
            if max_frames >= 2:
                indices.insert(0, max(0, center_idx - offset))
            if max_frames >= 3:
                indices.append(center_idx + offset)
            indices = indices[:max_frames]

        for idx in indices:
            frame = self._safe_seek(idx)
            if frame is None:
                continue
            b64 = self._frame_to_base64(frame)
            if b64:
                ts = round(idx / fps, 2) if fps > 0 else timestamp_sec
                results.append({
                    "timestamp_sec": ts,
                    "image_base64": b64,
                })

        # Если ничего не вышло — пробуем по времени
        if not results:
            frame = self.video_reader.seek_time(timestamp_sec)
            if frame is not None:
                b64 = self._frame_to_base64(frame)
                if b64:
                    results.append({
                        "timestamp_sec": timestamp_sec,
                        "image_base64": b64,
                    })

        # Гарантируем хотя бы один кадр
        if not results:
            logger.warning("Не удалось извлечь ни одного кадра для события @%s", timestamp_sec)

        return results

    def _safe_seek(self, frame_idx: int) -> Optional[np.ndarray]:
        """Безопасный переход к кадру."""
        try:
            frame = self.video_reader.seek_frame(frame_idx)
            return frame
        except Exception as e:
            logger.debug("Ошибка seek к кадру %d: %s", frame_idx, e)
            return None

    @staticmethod
    def _frame_to_base64(frame: np.ndarray, quality: int = 75) -> Optional[str]:
        """
        Конвертирует кадр OpenCV в JPEG Base64.
        """
        try:
            # OpenCV BGR → PIL RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # Уменьшаем если кадр слишком большой
            max_dim = 1280
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return b64
        except Exception as e:
            logger.error("Ошибка конвертации кадра в Base64: %s", e)
            return None
