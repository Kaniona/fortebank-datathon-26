"""
ForteBank Datathon 26 — Video Analyzer
FastAPI приложение: API + Jinja2 веб-интерфейс.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Не ставим пакеты и не обращаемся к сети во время работы: сервис должен быть переносимым
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.deps import valid_job_id
from app.config import HOST, PORT, ALLOWED_EXTENSIONS, MAX_FILE_SIZE, UPLOAD_DIR, API_KEY, JOB_RETENTION_HOURS
from app.database import init_db

# ── Логирование ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fortebank")


def _recover_state() -> None:
    """При старте: прерванные задачи помечаем failed, осиротевшие загрузки старше суток удаляем."""
    from app.database import SessionLocal
    from app.models.job import Job
    db = SessionLocal()
    try:
        stale = db.query(Job).filter(Job.status == "processing").all()
        for job in stale:
            job.status = "failed"
            job.error_message = "Обработка прервана перезапуском сервиса."
        # Приватность: результаты (в них кадры из видео) хранятся не дольше JOB_RETENTION_HOURS
        from datetime import datetime, timedelta
        cutoff_dt = datetime.utcnow() - timedelta(hours=JOB_RETENTION_HOURS)
        old = db.query(Job).filter(Job.created_at < cutoff_dt).delete(synchronize_session=False)
        db.commit()
        if stale:
            logger.warning("Помечено как failed прерванных задач: %d", len(stale))
        if old:
            logger.info("Удалено устаревших результатов: %d", old)
    finally:
        db.close()
    cutoff = time.time() - 24 * 3600
    for p in Path(UPLOAD_DIR).glob("*"):
        try:
            if p.is_file() and p.name != ".gitkeep" and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _recover_state()
    if not API_KEY:
        logger.warning("API_KEY не задан в .env — все запросы к API будут отклоняться (401).")
    if os.getenv("FORTEBANK_SKIP_WARMUP") != "1":
        # Модели загружаем заранее в фоне: первое видео жюри не должно платить за загрузку весов
        import asyncio
        from app.services.analysis_pipeline import warm_up
        asyncio.get_running_loop().run_in_executor(None, warm_up)
    logger.info("Сервис запущен")
    yield


# ── FastAPI ───────────────────────────────────────────────────
# Swagger/ReDoc подгружают скрипты с CDN (нужен интернет) — отключены; описание API в README.
app = FastAPI(
    title="ForteBank Datathon 26 — Video Analyzer",
    description=(
        "Локальный AI-сервис для выявления нарушений кассовой дисциплины "
        "по видеозаписям. Все модели работают локально, без облачных API."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# ── Заголовки безопасности ───────────────────────────────────
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for k, v in _SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("Необработанная ошибка: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса."})


# ── Статика и шаблоны ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── API роутеры ────────────────────────────────────────────────
from app.api.analyze import router as analyze_router
from app.api.jobs import router as jobs_router

app.include_router(analyze_router)
app.include_router(jobs_router)


# ── Веб-интерфейс (Jinja2) ────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница — загрузка видео."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "ForteBank Video Analyzer",
            "allowed_ext": ", ".join(sorted(ALLOWED_EXTENSIONS)),
            "max_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        },
    )


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    """Страница результатов анализа."""
    if not valid_job_id(job_id):
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return templates.TemplateResponse(
        request,
        "results.html",
        {"title": "Талдау нәтижесі", "job_id": job_id},
    )


@app.get("/api/v1/health")
async def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
