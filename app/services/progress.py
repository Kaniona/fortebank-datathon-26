"""Прогресс обработки задач в памяти процесса (не пишется в БД)."""
import threading

_lock = threading.Lock()
_state: dict[str, dict] = {}


def set_progress(job_id: str, percent: float, stage: str = "") -> None:
    with _lock:
        _state[job_id] = {"progress": int(max(0, min(100, percent))), "stage": stage}


def get_progress(job_id: str) -> dict:
    with _lock:
        return dict(_state.get(job_id, {"progress": 0, "stage": ""}))


def clear_progress(job_id: str) -> None:
    with _lock:
        _state.pop(job_id, None)
