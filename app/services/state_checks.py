"""
Проверки состояний, которые видны по пикселям: экран ноутбука (заблокирован ли ПК).
"""
from typing import Optional

import cv2
import numpy as np

UNLOCKED_MIN_V = 125      # средняя яркость (HSV-V) области экрана включённого ноутбука
DIM_DROP = 0.85           # падение яркости ниже 85% от максимума = экран заблокирован/погас


def laptop_brightness(frame_bgr: np.ndarray, box: list) -> Optional[float]:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return None
    return float(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2].mean())


def pc_left_unlocked(series: list[tuple[float, float]], t_leave: float) -> Optional[dict]:
    """
    series — [(время, яркость экрана)]. Возвращает описание, если кассир ушёл, а экран остался
    ярким и ни разу не гас/не блокировался до ухода; иначе None.
    """
    if not series:
        return None
    hist = [(t, v) for t, v in series if t <= t_leave + 0.5]
    near = [v for t, v in series if abs(t - t_leave) <= 1.5]
    if not hist or not near:
        return None
    runmax = max(v for _, v in hist)
    dim_before = any(v < DIM_DROP * runmax for _, v in hist)
    now = float(np.mean(near))
    if now >= UNLOCKED_MIN_V and not dim_before:
        return {"brightness": round(now, 1), "max_before": round(runmax, 1)}
    return None
