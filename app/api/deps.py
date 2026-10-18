"""
Зависимости и утилиты для API — авторизация, защита от перебора, проверка файлов.
"""
import os
import re
import secrets
import time
from collections import deque
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from app.config import (
    API_KEY, ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MAX_VIDEO_PIXELS, MAX_VIDEO_SECONDS,
    AUTH_FAIL_LIMIT, AUTH_FAIL_WINDOW_SEC,
)

JOB_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")

# ── Защита от перебора ключа: неудачные попытки на клиента ───────
_fail_log: dict[str, deque] = {}
_MAX_TRACKED_CLIENTS = 5000


def _client_id(request: Request) -> str:
    """IP клиента. За Cloudflare-туннелем реальный адрес приходит в CF-Connecting-IP."""
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


def _check_rate(client: str) -> None:
    now = time.monotonic()
    q = _fail_log.get(client)
    if q is None:
        return
    while q and now - q[0] > AUTH_FAIL_WINDOW_SEC:
        q.popleft()
    if len(q) >= AUTH_FAIL_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Слишком много неудачных попыток авторизации. Повторите позже.",
            headers={"Retry-After": str(AUTH_FAIL_WINDOW_SEC)},
        )


def _register_failure(client: str) -> None:
    if len(_fail_log) >= _MAX_TRACKED_CLIENTS:
        _fail_log.clear()  # защита памяти от переполнения при распределённом переборе
    _fail_log.setdefault(client, deque()).append(time.monotonic())


def reset_auth_throttle() -> None:
    """Сбрасывает счётчики неудачных попыток (для тестов)."""
    _fail_log.clear()


def verify_api_key(request: Request, x_api_key: Optional[str] = Header(None)):
    """
    Проверяет API-ключ в заголовке X-API-Key (сравнение за постоянное время).
    Если ключ в конфигурации не задан — отклоняет всё (fail-closed).
    """
    client = _client_id(request)
    _check_rate(client)
    ok = bool(API_KEY) and bool(x_api_key) and secrets.compare_digest(
        x_api_key.encode("utf-8"), API_KEY.encode("utf-8")
    )
    if not ok:
        _register_failure(client)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный или отсутствующий API-ключ. Укажите заголовок X-API-Key.",
        )
    return x_api_key


def validate_file_extension(filename: str) -> bool:
    """Проверяет, что файл имеет допустимое расширение."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """Проверяет, что размер файла не превышает лимит."""
    return file_size <= MAX_FILE_SIZE


def valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_RE.match(job_id or ""))


def generate_safe_filename(original_name: str) -> str:
    """
    Генерирует безопасное имя файла: случайный префикс + допустимое расширение.
    Исходное имя никогда не используется в пути (защита от path traversal).
    """
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".mov"
    return f"{secrets.token_hex(8)}{ext}"


def has_video_signature(file_path: str) -> bool:
    """Быстрая проверка сигнатуры контейнера (MOV/MP4, AVI, MKV/WebM) до открытия декодером."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False
    if len(head) < 12:
        return False
    if head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"):
        return True                      # QuickTime / MP4
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True                      # AVI
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return True                      # Matroska / WebM
    return False


def inspect_video(file_path: str) -> tuple[bool, str]:
    """
    Проверяет, что файл — реальное, декодируемое видео в разумных пределах.
    Возвращает (ok, сообщение_об_ошибке).
    """
    if not has_video_signature(file_path):
        return False, "Файл не похож на видео (неверная сигнатура контейнера)."
    import cv2
    cap = cv2.VideoCapture(file_path)
    try:
        if not cap.isOpened():
            return False, "Не удалось открыть видео — файл повреждён или формат не поддерживается."
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS); n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if w <= 0 or h <= 0 or w * h > MAX_VIDEO_PIXELS:
            return False, "Недопустимое разрешение видео."
        if fps and fps > 0 and n and n / fps > MAX_VIDEO_SECONDS:
            return False, f"Видео слишком длинное (максимум {MAX_VIDEO_SECONDS} с)."
        ok, _ = cap.read()
        if not ok:
            return False, "Не удалось прочитать кадры видео."
        return True, ""
    finally:
        cap.release()


def is_valid_video(file_path: str) -> bool:
    """Обратная совместимость: True, если inspect_video не нашёл проблем."""
    return inspect_video(file_path)[0]
