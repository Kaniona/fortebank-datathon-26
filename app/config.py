"""
Конфигурация приложения ForteBank Datathon 26.
Все настройки загружаются из переменных окружения (.env).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Базовые пути ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── API ───────────────────────────────────────────────────────
# Пустой ключ = сервис отклоняет все запросы (fail-closed). Значения-заглушки тоже считаются пустыми.
_PLACEHOLDER_KEYS = {"", "change-me-to-a-secure-key", "your_generated_key", "ваш_сгенерированный_ключ"}
API_KEY: str = os.getenv("API_KEY", "").strip()
if API_KEY in _PLACEHOLDER_KEYS:
    API_KEY = ""
MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(200 * 1024 * 1024)))  # 200 МБ (в кейсе ~29 МБ)

# ── Каталоги ──────────────────────────────────────────────────
UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
EVIDENCE_DIR: Path = BASE_DIR / os.getenv("EVIDENCE_DIR", "evidence")
MODEL_DIR: Path = BASE_DIR / os.getenv("MODEL_DIR", "models")
DB_PATH: Path = BASE_DIR / os.getenv("DB_PATH", "fortebank_datathon.db")

# Гарантируем существование каталогов
for d in (UPLOAD_DIR, EVIDENCE_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Обработка видео ──────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
FRAME_SAMPLE_STEP: int = int(os.getenv("FRAME_SAMPLE_STEP", "3"))
# Жёсткий бюджет на видео от момента получения (по кейсу — 5 минут; оставляем запас на выдачу ответа)
MAX_PROCESSING_TIME: int = int(os.getenv("MAX_PROCESSING_TIME", "270"))
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")

# ── Ограничения на входное видео (защита от «бомб» и перегрузки) ─
MAX_VIDEO_SECONDS: int = int(os.getenv("MAX_VIDEO_SECONDS", "900"))
MAX_VIDEO_PIXELS: int = int(os.getenv("MAX_VIDEO_PIXELS", str(4096 * 4096)))

# ── Хранение результатов (кадры из видео) — приватность ──────
JOB_RETENTION_HOURS: int = int(os.getenv("JOB_RETENTION_HOURS", "48"))

# ── Защита от перебора ключа ──────────────────────────────────
AUTH_FAIL_LIMIT: int = int(os.getenv("AUTH_FAIL_LIMIT", "10"))      # неудач за окно
AUTH_FAIL_WINDOW_SEC: int = int(os.getenv("AUTH_FAIL_WINDOW_SEC", "60"))

# ── Сервер ────────────────────────────────────────────────────
# По умолчанию слушаем только localhost: наружу сервис публикуется через туннель.
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))

# ── Разрешённые форматы ───────────────────────────────────────
ALLOWED_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}
