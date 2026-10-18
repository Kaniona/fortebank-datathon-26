"""
Поведенческий анализ кассовой сцены по трекам людей (без привязки к конкретной камере).

Все пространственные величины измеряются в высотах кассира (H), поэтому ракурс и масштаб
влияют слабо. Функции чистые: принимают записи кадров Perceiver и возвращают структуры.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.services.tracks import Track, build_tracks, merge_fragments, real_tracks, pick_cashier

AWAY_DIST_H = 0.6        # дальше этого расстояния (в высотах кассира) от «якоря» (места за столом) кассир считается отошедшим
AWAY_MIN_SEC = 3.0       # минимальная длительность отсутствия, чтобы считать это уходом
SEATED_SPEED_H = 0.25    # скорость (H/с) ниже которой человек считается неподвижным


@dataclass
class Scene:
    duration: float
    cashier: Optional[Track]
    H: float = 0.0
    anchor: tuple = (0.0, 0.0)
    desk_roi: Optional[list] = None
    away: list = field(default_factory=list)       # [(t_start, t_end)] — кассир встал и затем покинул зону стола
    vacant_from: dict = field(default_factory=dict)  # {(t_start, t_end): время, с которого стол пуст}
    real: list = field(default_factory=list)
    ghosts: list = field(default_factory=list)
    laptop_box: Optional[list] = None


def _speed(track: Track) -> np.ndarray:
    c = track.centers(); t = np.array(track.ts)
    if len(t) < 2:
        return np.zeros(len(t))
    sp = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3) / max(track.height, 1.0)
    return np.concatenate([[sp[0]], sp])


def desk_roi_for(cashier: Track, anchor, H, frame_w, frame_h) -> list:
    """Зона рабочего стола: бокс кассира + след запястий, расширенные на 0.4H."""
    xs, ys = [], []
    for p in cashier.persons:
        x1, y1, x2, y2 = p["box"]
        xs += [x1, x2]; ys += [y1, y2]
        for k in (9, 10):                              # запястья
            x, y, c = p["kpts"][k]
            if c > 0.3:
                xs.append(x); ys.append(y)
    # робастные границы, чтобы единичные выбросы не раздували зону
    x1, x2 = np.percentile(xs, [3, 97]); y1, y2 = np.percentile(ys, [3, 97])
    pad = 0.4 * H
    return [int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(frame_w, x2 + pad)), int(min(frame_h, y2 + pad))]


def _vacant_flags(frames: list[dict], roi: list, min_height: float) -> list[bool]:
    """Для каждого кадра: True, если в зоне стола нет человека (учитываются только заметные по размеру люди)."""
    x1, y1, x2, y2 = roi
    out = []
    for f in frames:
        busy = False
        for p in f["persons"]:
            b = p["box"]
            if b[3] - b[1] < min_height:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                busy = True
                break
        out.append(not busy)
    return out


def analyze_scene(frames: list[dict], duration: float, frame_w: int, frame_h: int) -> Scene:
    tracks = merge_fragments(build_tracks(frames))
    real, ghosts = real_tracks(tracks)
    lap = [o["box"] for f in frames for o in f["objs"] if o["cls"] == "laptop" and o["conf"] >= 0.5]
    lap_box = [int(v) for v in np.median(np.array(lap), axis=0)] if lap else None
    cashier = pick_cashier(real, lap_box)
    sc = Scene(duration=duration, cashier=cashier, real=real, ghosts=ghosts)
    if cashier is None:
        return sc

    # Сидит/стоит: у сидящего за столом бокс примерно вдвое ниже, чем у стоящего
    h = cashier.heights(); h_stand = float(np.percentile(h, 90)); h_low = float(np.percentile(h, 10))
    both = h_stand / max(h_low, 1.0) > 1.5
    seated = (h < 0.72 * h_stand) if both else np.ones(len(h), bool)
    sc.H = h_stand
    c = cashier.centers()
    sc.anchor = tuple(np.median(c[seated], axis=0)) if seated.sum() >= 3 else tuple(np.median(c, axis=0))
    sub = Track(cashier.id, [t for t, k in zip(cashier.ts, seated) if k], [p for p, k in zip(cashier.persons, seated) if k])
    sc.desk_roi = desk_roi_for(sub if len(sub.ts) >= 3 else cashier, sc.anchor, sc.H, frame_w, frame_h)

    lap = [o["box"] for f in frames for o in f["objs"] if o["cls"] == "laptop" and o["conf"] >= 0.5]
    if lap:
        sc.laptop_box = [int(v) for v in np.median(np.array(lap), axis=0)]

    # Интервалы, когда кассир не сидит; «ушёл» — если после этого зона стола опустела
    ts = [f["t"] for f in frames]
    seat_by_t = {t: bool(k) for t, k in zip(cashier.ts, seated)}
    vacant = _vacant_flags(frames, sc.desk_roi, 0.45 * sc.H)
    not_seated_flags = []
    for t in ts:
        near = [seat_by_t[x] for x in cashier.ts if abs(x - t) <= 1.2]
        not_seated_flags.append(True if not near else (sum(near) / len(near) < 0.5))
    i = 0
    while i < len(ts):
        if not_seated_flags[i]:
            j = i
            while j + 1 < len(ts) and not_seated_flags[j + 1]:
                j += 1
            t0, t1 = ts[i], ts[j]
            end = max(t1, duration) if j == len(ts) - 1 else t1
            if end - t0 >= AWAY_MIN_SEC and t0 >= 1.0:
                vt = next((ts[k] for k in range(i, len(ts)) if vacant[k] and (k + 1 >= len(ts) or vacant[k + 1] or k + 1 > j)), None)
                if vt is not None:
                    sc.away.append((t0, end)); sc.vacant_from[(t0, end)] = vt
            i = j + 1
        else:
            i += 1
    return sc
