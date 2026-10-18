"""
Pydantic-схемы для API-контракта ForteBank Datathon 26.
"""
from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeResponse(BaseModel):
    """Ответ POST /api/v1/analyze."""
    job_id: str = Field(..., description="Уникальный идентификатор задачи")
    status: str = Field("processing", description="Статус задачи: processing")


class EvidenceFrame(BaseModel):
    """Доказательный кадр."""
    timestamp_sec: float = Field(..., description="Время кадра в секундах")
    image_base64: str = Field(..., description="JPEG-кадр в Base64")


class ViolationItem(BaseModel):
    """Описание обнаруженного нарушения."""
    type: str = Field(..., description="Тип предполагаемого нарушения")
    timecode_sec: float = Field(..., description="Таймкод события в секундах")
    description: str = Field(..., description="Описание визуально наблюдаемого действия")
    confidence: float = Field(0.0, description="Оценка уверенности модели (0.0–1.0)")
    evidence_frames: list[EvidenceFrame] = Field(
        default_factory=list,
        description="1–3 JPEG-кадра в Base64"
    )
    regulatory_reference: Optional[str] = Field(
        None, description="Нормативный источник (если подтверждён)"
    )
    limitations: Optional[str] = Field(
        None, description="Ограничения анализа для данного события"
    )


class JobResultResponse(BaseModel):
    """Ответ GET /api/v1/jobs/{job_id}."""
    job_id: str
    status: str  # processing | done | failed
    verdict: Optional[str] = None  # violation | no_violation
    violations: list[ViolationItem] = Field(default_factory=list)
    error: Optional[str] = None
