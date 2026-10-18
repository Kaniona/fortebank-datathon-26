"""
«Дверь кассового помещения оставлена открытой».

Идея: камера неподвижна. Берём эталон сцены из первых кадров и ищем в верхней части кадра (где дверь)
крупную вертикальную область, которая изменилась относительно эталона и не объясняется людьми
(боксы людей ±1.5 с замаскированы). Если такая область держится несколько секунд — полотно двери
осталось в другом положении (открыто). Смена освещения во всём кадре отбрасывается.
Все величины — доли размера кадра, поэтому от разрешения и ракурса зависят слабо.
"""
from typing import Optional

import cv2
import numpy as np

SCALE = 0.5            # анализ на уменьшенном кадре
DIFF_THR = 25          # порог разницы яркости
TOP = 0.55             # анализируем верхние 55% кадра
MASK_WIN = 1.5         # маскируем людей в окне ±1.5 с
MASK_PAD = 0.2
MIN_H, MIN_W = 0.30, 0.10   # «полотно двери»: высота ≥30% и ширина ≥10% кадра
LEVEL = 0.02           # площадь такой области ≥2% кадра
HOLD, FRAC = 5.0, 0.7  # держится ≥70% времени в окне 5 с
GLOBAL_MAX = 0.4       # если изменилось >40% кадра — это свет, а не дверь


def prep(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (5, 5), 0)


def _person_mask(rec: dict, shape) -> np.ndarray:
    H, W = shape
    m = np.zeros(shape, bool)
    for p in rec["persons"]:
        x1, y1, x2, y2 = [v * SCALE for v in p["box"]]
        w, h = x2 - x1, y2 - y1
        m[max(0, int(y1 - MASK_PAD * h)):int(y2 + MASK_PAD * h), max(0, int(x1 - MASK_PAD * w)):int(x2 + MASK_PAD * w)] = True
    return m


def door_timeline(grays: list[np.ndarray], frames: list[dict]) -> list[tuple[float, float, float]]:
    """grays[i] соответствует frames[i]. Возвращает [(t, площадь «двери», доля глобальных изменений)]."""
    if len(grays) < 6:
        return []
    shape = grays[0].shape; H, W = shape; y2 = int(H * TOP)
    pm = [_person_mask(f, shape) for f in frames]
    ts = [f["t"] for f in frames]
    st = np.stack([g.astype(np.float32) for g in grays[:4]]); ms = np.stack(pm[:4])
    ref = np.ma.median(np.ma.array(st, mask=ms), axis=0).filled(0); known = ~ms.all(0)
    kern = np.ones((5, 5), np.uint8)
    out = []
    for i, (t, g) in enumerate(zip(ts, grays)):
        diff = np.abs(g.astype(np.float32) - ref) > DIFF_THR
        gfrac = float(diff[known].mean()) if known.any() else 0.0
        m = known.copy()
        for j in range(i, -1, -1):
            if t - ts[j] > MASK_WIN: break
            m &= ~pm[j]
        for j in range(i + 1, len(ts)):
            if ts[j] - t > MASK_WIN: break
            m &= ~pm[j]
        m[y2:] = False
        bm = cv2.morphologyEx(((diff & m).astype(np.uint8)) * 255, cv2.MORPH_CLOSE, kern)
        n, _, stats, _ = cv2.connectedComponentsWithStats(bm)
        best = 0.0; box = None
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            if h >= MIN_H * H and w >= MIN_W * W and a / (H * W) > best:
                best = a / (H * W); box = [int(x / SCALE), int(y / SCALE), int((x + w) / SCALE), int((y + h) / SCALE)]
        out.append((t, best, gfrac, box))
    return out


def door_left_open(tl) -> Optional[dict]:
    """Первый момент, после которого «полотно двери» держится в изменённом положении."""
    tl = [x for x in tl if x[2] < GLOBAL_MAX]
    if not tl:
        return None
    ts = np.array([x[0] for x in tl]); v = np.array([x[1] for x in tl])
    for i, t in enumerate(ts):
        if v[i] < LEVEL:
            continue
        if ts[-1] - t < HOLD * 0.8:
            break
        w = (ts >= t) & (ts <= t + HOLD)
        if (v[w] >= LEVEL).mean() >= FRAC:
            j = int(np.argmax(np.where(w, v, -1)))
            return {"t": float(t), "area": round(float(np.median(v[w])), 3), "box": tl[j][3], "evidence_t": float(ts[j])}
    return None
