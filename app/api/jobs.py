"""
GET /api/v1/jobs/{job_id} — получение статуса и результата задачи.
Также содержит фоновую функцию обработки видео.
"""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import verify_api_key, valid_job_id
from app.config import MAX_PROCESSING_TIME
from app.database import SessionLocal
from app.models.job import Job
from app.services.progress import set_progress, clear_progress

logger = logging.getLogger(__name__)

router = APIRouter()

# Анализ — тяжёлая CPU-задача: выполняем в отдельном потоке (не блокируем event loop)
# и по одному видео за раз.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")

# Сообщение клиенту при внутренней ошибке: детали пишем только в лог (без утечки путей и трассировок)
GENERIC_ERROR = "Не удалось обработать видео. Попробуйте другой файл или повторите позже."


def _deadline_for(job: Job) -> float:
    """Крайний срок ответа считается от момента получения видео, а не от старта обработки."""
    from datetime import timezone
    created = job.created_at.replace(tzinfo=timezone.utc).timestamp()
    return created + MAX_PROCESSING_TIME


async def process_video_background(job_id: str):
    """Фоновая задача: анализирует видео и обновляет запись в БД."""
    logger.info("Начата фоновая обработка задачи %s", job_id)
    set_progress(job_id, 1, "Кезекте")

    db = SessionLocal()
    file_path = None
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error("Задача %s не найдена в БД", job_id)
            return
        file_path = job.file_path
        deadline = _deadline_for(job)

        try:
            from app.services.analysis_pipeline import AnalysisPipeline

            pipeline = AnalysisPipeline()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: pipeline.analyze(
                    file_path, deadline=deadline,
                    progress=lambda p, s: set_progress(job_id, p, s),
                ),
            )

            job.status = "done"
            job.verdict = result["verdict"]
            job.violations_json = json.dumps(result["violations"], ensure_ascii=False)
            vi = result.get("video_info", {})
            job.processing_sec = str(vi.get("processing_sec", ""))
            job.duration_sec = str(vi.get("duration_sec", ""))
            job.fps = str(vi.get("fps", ""))
            job.resolution = vi.get("resolution", "")
            db.commit()
            logger.info("Задача %s завершена: вердикт=%s, нарушений=%d",
                        job_id, result["verdict"], len(result["violations"]))

        except Exception:
            logger.exception("Ошибка обработки задачи %s", job_id)
            db.rollback()
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = GENERIC_ERROR
                db.commit()
    finally:
        db.close()
        clear_progress(job_id)
        # Видео заказчика не храним дольше необходимого: результат и кадры уже в БД
        if file_path:
            try:
                import os
                os.remove(file_path)
            except OSError:
                pass


@router.get("/api/v1/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Возвращает статус и результат задачи.
    При status=done — включает вердикт и список нарушений.
    При status=failed — включает описание ошибки.
    """
    if not valid_job_id(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена.")
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена.")
        return job.to_dict()
    finally:
        db.close()
