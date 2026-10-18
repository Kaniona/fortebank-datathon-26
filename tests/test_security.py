"""Тесты безопасности и устойчивости API."""
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import init_db

init_db()
client = TestClient(app)


def _mp4_bytes(tmp_path):
    import cv2, numpy as np
    p = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(10):
        w.write(np.full((120, 160, 3), 90, np.uint8))
    w.release()
    return open(p, "rb").read()


class TestHeaders:
    def test_security_headers_present(self):
        r = client.get("/")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert "default-src 'self'" in r.headers["content-security-policy"]
        assert "script-src 'self'" in r.headers["content-security-policy"]

    def test_api_not_cached(self, auth):
        r = client.get("/api/v1/health")
        assert r.headers.get("cache-control") == "no-store"

    def test_docs_disabled(self):
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404


class TestAuth:
    def test_bruteforce_is_throttled(self):
        for _ in range(5):
            assert client.get("/api/v1/jobs/abcdef01", headers={"X-API-Key": "wrong"}).status_code == 401
        r = client.get("/api/v1/jobs/abcdef01", headers={"X-API-Key": "wrong"})
        assert r.status_code == 429 and "retry-after" in r.headers

    def test_valid_key_after_failures_is_also_throttled_then_recovers(self, auth):
        from app.api.deps import reset_auth_throttle
        for _ in range(5):
            client.get("/api/v1/jobs/abcdef01", headers={"X-API-Key": "wrong"})
        assert client.get("/api/v1/jobs/abcdef01", headers=auth).status_code == 429
        reset_auth_throttle()
        assert client.get("/api/v1/jobs/abcdef01", headers=auth).status_code == 404

    def test_fail_closed_when_key_not_configured(self, monkeypatch):
        import app.api.deps as deps
        monkeypatch.setattr(deps, "API_KEY", "")
        assert client.get("/api/v1/jobs/abcdef01", headers={"X-API-Key": ""}).status_code == 401
        assert client.get("/api/v1/jobs/abcdef01", headers={"X-API-Key": "anything"}).status_code == 401

    def test_placeholder_key_is_treated_as_unset(self, monkeypatch):
        import importlib, app.config as cfg
        monkeypatch.setenv("API_KEY", "change-me-to-a-secure-key")
        importlib.reload(cfg)
        try:
            assert cfg.API_KEY == ""
        finally:
            monkeypatch.setenv("API_KEY", "test-key-for-pytest-only-0123456789abcdef")
            importlib.reload(cfg)


class TestJobIds:
    @pytest.mark.parametrize("bad", ["../etc/passwd", "ABCDEF01", "zz", "a" * 40, "1; DROP TABLE jobs"])
    def test_invalid_job_id_returns_404(self, auth, bad):
        assert client.get(f"/api/v1/jobs/{bad}", headers=auth).status_code in (404, 422)

    def test_results_page_rejects_bad_id(self):
        assert client.get("/results/not-a-valid-id").status_code == 404
        assert client.get("/results/<script>alert(1)</script>").status_code == 404

    def test_results_page_valid_id(self):
        assert client.get("/results/0123456789abcdef").status_code == 200


class TestUploadValidation:
    def test_text_file_with_mov_extension_rejected(self, auth):
        r = client.post("/api/v1/analyze", headers=auth,
                        files={"file": ("fake.mov", io.BytesIO(b"this is definitely not a video file" * 50), "video/quicktime")})
        assert r.status_code == 400

    def test_empty_file_rejected(self, auth):
        r = client.post("/api/v1/analyze", headers=auth, files={"file": ("e.mov", io.BytesIO(b""), "video/quicktime")})
        assert r.status_code == 400

    def test_oversize_rejected_and_cleaned(self, auth, monkeypatch):
        import app.api.analyze as an
        from app.config import UPLOAD_DIR
        monkeypatch.setattr(an, "MAX_FILE_SIZE", 1024)
        before = set(UPLOAD_DIR.glob("*"))
        r = client.post("/api/v1/analyze", headers=auth,
                        files={"file": ("big.mov", io.BytesIO(b"\x00" * 5000), "video/quicktime")})
        assert r.status_code == 413
        assert set(UPLOAD_DIR.glob("*")) == before

    def test_path_traversal_filename_is_harmless(self, auth, tmp_path):
        from app.config import UPLOAD_DIR
        r = client.post("/api/v1/analyze", headers=auth,
                        files={"file": ("../../evil.mp4", io.BytesIO(_mp4_bytes(tmp_path)), "video/mp4")})
        assert r.status_code == 202
        assert not (UPLOAD_DIR.parent.parent / "evil.mp4").exists()

    def test_uploaded_video_deleted_after_processing(self, auth, tmp_path):
        from app.config import UPLOAD_DIR
        with TestClient(app) as c:      # контекст запускает lifespan и event loop для фоновой задачи
            r = c.post("/api/v1/analyze", headers=auth,
                       files={"file": ("ok.mp4", io.BytesIO(_mp4_bytes(tmp_path)), "video/mp4")})
            assert r.status_code == 202
            job_id = r.json()["job_id"]
            assert len(job_id) == 16
            import time
            for _ in range(50):
                d = c.get(f"/api/v1/jobs/{job_id}", headers=auth).json()
                if d["status"] != "processing":
                    break
                time.sleep(0.1)
            assert d["status"] == "done" and d["verdict"] == "no_violation" and d["violations"] == []
            from app.database import SessionLocal
            from app.models.job import Job
            import os
            db = SessionLocal()
            try:
                path = db.query(Job).filter(Job.job_id == job_id).first().file_path
            finally:
                db.close()
            assert not os.path.exists(path), "загруженное видео должно удаляться после обработки"


class TestErrorSanitizing:
    def test_internal_errors_do_not_leak_details(self, auth, tmp_path, monkeypatch):
        import app.services.analysis_pipeline as ap
        class Boom:
            def analyze(self, *a, **k):
                raise RuntimeError("/home/secret/path/model.pt not found")
        monkeypatch.setattr(ap, "AnalysisPipeline", Boom)
        import time
        with TestClient(app) as c:
            r = c.post("/api/v1/analyze", headers=auth,
                       files={"file": ("ok.mp4", io.BytesIO(_mp4_bytes(tmp_path)), "video/mp4")})
            job_id = r.json()["job_id"]
            for _ in range(50):
                d = c.get(f"/api/v1/jobs/{job_id}", headers=auth).json()
                if d["status"] != "processing":
                    break
                time.sleep(0.1)
            assert d["status"] == "failed"
            assert "secret" not in str(d) and "model.pt" not in str(d)


class TestRetention:
    def test_old_results_are_purged_on_startup(self):
        from datetime import datetime, timedelta
        from app.database import SessionLocal
        from app.models.job import Job
        from app.main import _recover_state
        db = SessionLocal()
        try:
            old = Job(filename="o.mov", original_filename="o.mov", file_path="/nonexistent/o.mov", status="done",
                      verdict="no_violation", violations_json="[]", created_at=datetime.utcnow() - timedelta(hours=100))
            new = Job(filename="n.mov", original_filename="n.mov", file_path="/nonexistent/n.mov", status="done",
                      verdict="no_violation", violations_json="[]")
            db.add_all([old, new]); db.commit()
            old_id, new_id = old.job_id, new.job_id
        finally:
            db.close()
        _recover_state()
        db = SessionLocal()
        try:
            assert db.query(Job).filter(Job.job_id == old_id).first() is None
            assert db.query(Job).filter(Job.job_id == new_id).first() is not None
        finally:
            db.close()
