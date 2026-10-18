"""
Оқшаулау: тесттер өнімдік .env, дерекқор және каталогтарды қолданбайды.
Айнымалылар app импортталғанға дейін орнатылуы тиіс.
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="fortebank_tests_"))
os.environ["API_KEY"] = "test-key-for-pytest-only-0123456789abcdef"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["EVIDENCE_DIR"] = str(_TMP / "evidence")
os.environ["DB_PATH"] = str(_TMP / "test.db")
os.environ["AUTH_FAIL_LIMIT"] = "5"
os.environ["FORTEBANK_SKIP_WARMUP"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


class FakePipeline:
    """Заглушка: тесты API не должны загружать нейросети."""
    verdict = "no_violation"

    def analyze(self, file_path, deadline=None, progress=None):
        if progress:
            progress(50, "test")
        return {
            "verdict": self.verdict, "violations": [],
            "video_info": {"resolution": "320x240", "fps": 10.0, "duration_sec": 2.0, "processing_sec": 0.1},
        }


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch):
    import app.services.analysis_pipeline as ap
    monkeypatch.setattr(ap, "AnalysisPipeline", FakePipeline)


@pytest.fixture(autouse=True)
def _reset_throttle():
    from app.api.deps import reset_auth_throttle
    reset_auth_throttle()
    yield
    reset_auth_throttle()


@pytest.fixture
def auth():
    from app.config import API_KEY
    return {"X-API-Key": API_KEY}
