"""
ORM-модель задачи анализа видео.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, Integer

from app.database import Base


def _generate_job_id() -> str:
    """Случайный идентификатор задачи (16 hex-символов, не угадывается)."""
    return uuid.uuid4().hex[:16]


class Job(Base):
    """Таблица задач анализа видео."""

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(32), unique=True, index=True, default=_generate_job_id, nullable=False)

    # Статус: processing | done | failed
    status = Column(String(20), default="processing", nullable=False)

    # Вердикт: violation | no_violation | null (пока не обработано)
    verdict = Column(String(20), nullable=True)

    # Исходные данные
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, default=0)
    duration_sec = Column(String(20), nullable=True)   # храним как строку для простоты
    fps = Column(String(20), nullable=True)
    resolution = Column(String(20), nullable=True)
    processing_sec = Column(String(20), nullable=True)

    # Результат анализа (JSON-строка)
    violations_json = Column(Text, nullable=True)

    # Метаданные
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Возвращает словарь для API-ответа."""
        import json
        data = {
            "job_id": self.job_id,
            "status": self.status,
        }
        if self.status == "processing":
            from app.services.progress import get_progress
            data.update(get_progress(self.job_id))
        if self.verdict is not None:
            data["verdict"] = self.verdict
        if self.status == "done":
            violations = []
            if self.violations_json:
                try:
                    violations = json.loads(self.violations_json)
                except (json.JSONDecodeError, TypeError):
                    violations = []
            data["violations"] = violations
            data["video_info"] = {
                "resolution": self.resolution, "fps": float(self.fps) if self.fps else None,
                "duration_sec": float(self.duration_sec) if self.duration_sec else None,
                "processing_sec": float(self.processing_sec) if self.processing_sec else None,
            }
        if self.status == "failed" and self.error_message:
            data["error"] = self.error_message
        return data
