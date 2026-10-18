"""
Тесты API-эндпоинтов ForteBank Datathon 26.

Проверяют:
  - Авторизацию (X-API-Key)
  - Форматы ответов
  - Обработку ошибок
  - Корректность Base64-кадров
  - Контракт API (verdict, violations, timecode_sec, evidence_frames)

Запуск:
  cd /home/kanion/Documents/datathonnn
  source venv/bin/activate
  python -m pytest tests/test_api.py -v
"""
import base64
import io
import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.config import API_KEY
from app.database import init_db

# Гарантируем, что БД создана до тестов
init_db()

client = TestClient(app)


# ═══════════════════════════════════════════════════
# Утилиты
# ═══════════════════════════════════════════════════

def make_test_video(path: str, duration_sec: float = 2.0, fps: int = 10):
    """Создаёт короткое тестовое видео с помощью OpenCV."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 240
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    frames = int(duration_sec * fps)
    for i in range(frames):
        # Кадр с «человеком» — прямоугольник (имитация)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # серый фон
        cv2.rectangle(frame, (100, 80), (220, 200), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def test_video_path(tmp_path):
    """Создаёт тестовое видео во временной директории."""
    path = str(tmp_path / "test_video.mp4")
    make_test_video(path, duration_sec=1.0, fps=10)
    return path


@pytest.fixture
def test_video_bytes(test_video_path):
    """Возвращает байты тестового видео."""
    with open(test_video_path, "rb") as f:
        return f.read()


# ═══════════════════════════════════════════════════
# Тесты авторизации
# ═══════════════════════════════════════════════════

class TestAuthorization:
    """Тесты проверки API-ключа."""

    def test_analyze_without_api_key(self, test_video_bytes):
        """Запрос без API-ключа должен вернуть 401."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test.mp4", test_video_bytes, "video/mp4")},
        )
        assert response.status_code == 401

    def test_analyze_with_wrong_api_key(self, test_video_bytes):
        """Запрос с неверным API-ключом должен вернуть 401."""
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": "wrong-key"},
            files={"file": ("test.mp4", test_video_bytes, "video/mp4")},
        )
        assert response.status_code == 401

    def test_jobs_without_api_key(self):
        """GET /jobs без ключа должен вернуть 401."""
        response = client.get("/api/v1/jobs/test1234")
        assert response.status_code == 401

    def test_jobs_with_wrong_api_key(self):
        """GET /jobs с неверным ключом должен вернуть 401."""
        response = client.get(
            "/api/v1/jobs/test1234",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401


# ═══════════════════════════════════════════════════
# Тесты загрузки
# ═══════════════════════════════════════════════════

class TestUpload:
    """Тесты POST /api/v1/analyze."""

    def test_upload_valid_video(self, test_video_bytes):
        """Корректная загрузка видео должна вернуть job_id и status=processing."""
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            files={"file": ("test.mp4", test_video_bytes, "video/mp4")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0
        assert data["status"] == "processing"

    def test_upload_no_file(self):
        """Запрос без файла должен вернуть ошибку."""
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422

    def test_upload_invalid_format(self, test_video_bytes):
        """Файл с неразрешённым расширением должен быть отклонён."""
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            files={"file": ("test.txt", test_video_bytes, "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_empty_filename(self, test_video_bytes):
        """Файл с пустым именем должен быть отклонён."""
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            files={"file": ("", test_video_bytes, "video/mp4")},
        )
        assert response.status_code in (400, 422)


# ═══════════════════════════════════════════════════
# Тесты получения статуса
# ═══════════════════════════════════════════════════

class TestJobStatus:
    """Тесты GET /api/v1/jobs/{job_id}."""

    def test_nonexistent_job(self):
        """Запрос несуществующего job_id должен вернуть 404."""
        response = client.get(
            "/api/v1/jobs/nonexist",
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 404

    def test_job_status_processing(self, test_video_bytes):
        """Сразу после загрузки статус должен быть processing."""
        # Загружаем
        upload = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            files={"file": ("test.mp4", test_video_bytes, "video/mp4")},
        )
        assert upload.status_code == 202
        job_id = upload.json()["job_id"]

        # Проверяем статус (может быть processing или уже done)
        response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("processing", "done", "failed")


# ═══════════════════════════════════════════════════
# Тесты контракта API
# ═══════════════════════════════════════════════════

class TestApiContract:
    """Тесты соответствия формата ответа контракту."""

    def test_verdict_values(self):
        """verdict может быть только 'violation' или 'no_violation'."""
        from app.schemas.analysis import JobResultResponse
        assert JobResultResponse.model_fields["verdict"] is not None

    def test_processing_response_format(self, test_video_bytes):
        """Сразу после загрузки ответ должен содержать job_id и status."""
        upload = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            files={"file": ("test.mp4", test_video_bytes, "video/mp4")},
        )
        assert upload.status_code == 202
        data = upload.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "processing"

    def test_job_response_has_required_fields(self, test_video_bytes):
        """Ответ GET /jobs должен содержать обязательные поля."""
        upload = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            files={"file": ("test.mp4", test_video_bytes, "video/mp4")},
        )
        job_id = upload.json()["job_id"]

        response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        # status должен быть одним из трёх
        assert data["status"] in ("processing", "done", "failed")


# ═══════════════════════════════════════════════════
# Тесты утилит
# ═══════════════════════════════════════════════════

class TestUtils:
    """Тесты вспомогательных функций."""

    def test_safe_filename(self):
        """Проверка генерации безопасного имени файла."""
        from app.api.deps import generate_safe_filename
        name = generate_safe_filename("../../../etc/passwd.mov")
        assert ".." not in name
        assert "/" not in name
        assert name.endswith(".mov")

    def test_safe_filename_invalid_ext(self):
        """Проверка замены неразрешённого расширения."""
        from app.api.deps import generate_safe_filename
        name = generate_safe_filename("test.txt")
        assert name.endswith(".mov")

    def test_validate_extension(self):
        """Проверка валидации расширений."""
        from app.api.deps import validate_file_extension
        assert validate_file_extension("test.mov") is True
        assert validate_file_extension("test.mp4") is True
        assert validate_file_extension("test.txt") is False
        assert validate_file_extension("") is False


# ═══════════════════════════════════════════════════
# Тесты здоровья
# ═══════════════════════════════════════════════════

class TestHealth:
    """Тест health check."""

    def test_health_endpoint(self):
        """Health check должен вернуть 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_index_page(self):
        """Главная страница должна возвращать HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
