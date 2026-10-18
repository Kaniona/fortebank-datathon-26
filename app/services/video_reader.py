"""
Чтение и проверка видеофайлов через OpenCV.
"""
import cv2
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoInfo:
    """Метаданные видеофайла."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"


class VideoReader:
    """Обёртка над cv2.VideoCapture с удобными методами."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self):
        """Открывает видео и проверяет корректность."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Видеофайл не найден: {self.file_path}")

        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            raise ValueError(
                f"Не удалось открыть видеофайл (возможно, повреждён): {self.file_path}"
            )

    def get_info(self) -> VideoInfo:
        """Возвращает метаданные видео."""
        cap = self.cap
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        # Кодек (четыре символа)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            codec=codec,
        )

    def read_frame(self):
        """Читает следующий кадр. Возвращает (frame_index, frame) или None."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return idx, frame

    def seek_frame(self, frame_index: int):
        """Перематывает к указанному кадру и читает его."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def seek_time(self, timestamp_sec: float):
        """Перематывает к указанному времени и читает кадр."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    @staticmethod
    def frame_to_timestamp(frame_index: int, fps: float) -> float:
        """Преобразует индекс кадра в секунды."""
        return round(frame_index / fps, 2) if fps > 0 else 0.0

    def release(self):
        """Освобождает ресурсы."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
