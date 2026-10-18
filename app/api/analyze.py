"""
POST /api/v1/analyze — загрузка видео для анализа.
"""
import asyncio
import logging
import os

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.deps import (
    verify_api_key,
    validate_file_extension,
    generate_safe_filename,
    inspect_video,
)
from app.config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from app.database import SessionLocal
from app.models.job import Job
from app.schemas.analysis import AnalyzeResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Ссылки на фоновые задачи: без них asyncio может собрать задачу сборщиком мусора до завершения
_background_tasks: set = set()


@router.post("/api/v1/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """
    Принимает видеофайл (.mov, .mp4, .avi, .mkv), создаёт задачу анализа
    и возвращает job_id. Обработка выполняется асинхронно в фоне.
    """
    # Быстрый отказ по заявленной длине тела — до чтения файла
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_FILE_SIZE + 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл слишком большой. Максимум: {MAX_FILE_SIZE // (1024 * 1024)} МБ.",
        )

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Имя файла не указано.")

    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый формат. Разрешённые: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    # ── Сохранение файла потоково, с контролем размера ──────
    safe_name = generate_safe_filename(file.filename)
    upload_path = UPLOAD_DIR / safe_name
    file_size = 0
    too_large = False
    try:
        with open(upload_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    too_large = True
                    break
                f.write(chunk)
    except Exception:
        upload_path.unlink(missing_ok=True)
        logger.exception("Ошибка сохранения файла")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка сохранения файла.")
    finally:
        await file.close()

    if too_large:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл слишком большой. Максимум: {MAX_FILE_SIZE // (1024 * 1024)} МБ.",
        )
    if file_size == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файл пустой.")

    # ── Проверка реального содержимого (в потоке: декодер синхронный) ──
    ok, reason = await asyncio.get_running_loop().run_in_executor(None, inspect_video, str(upload_path))
    if not ok:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=reason)

    # ── Создание задачи ──────────────────────────────────────
    db = SessionLocal()
    try:
        job = Job(
            filename=safe_name,
            original_filename=os.path.basename(file.filename)[:255],
            file_path=str(upload_path),
            file_size=file_size,
            status="processing",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.job_id
    except Exception:
        upload_path.unlink(missing_ok=True)
        logger.exception("Не удалось создать задачу")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось создать задачу.")
    finally:
        db.close()

    from app.api.jobs import process_video_background
    task = asyncio.create_task(process_video_background(job_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.info("Создана задача %s (%d байт)", job_id, file_size)
    return AnalyzeResponse(job_id=job_id, status="processing")
