"""
Треки людей по записям кадров: фильтрация «людей за стеклом», склейка разорванных ID,
выбор роли кассира. Работает с записями Perceiver.process() (dict), без привязки к камере.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# keypoints COCO-17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = range(17)


@dataclass
class Track:
    id: int
    ts: list = field(default_factory=list)       # время, сек
    persons: list = field(default_factory=list)  # dict человека на каждом отсчёте

    @property
    def t0(self): return self.ts[0]
    @property
    def t1(self): return self.ts[-1]
    @property
    def duration(self): return self.ts[-1] - self.ts[0]

    def heights(self): return np.array([p["box"][3] - p["box"][1] for p in self.persons])
    def centers(self): return np.array([((p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2) for p in self.persons])
    @property
    def height(self): return float(np.median(self.heights()))

    def at(self, t, tol=0.8) -> Optional[dict]:
        """Ближайший отсчёт по времени (в пределах tol), иначе None."""
        i = int(np.argmin(np.abs(np.array(self.ts) - t)))
        return self.persons[i] if abs(self.ts[i] - t) <= tol else None


def build_tracks(frames, min_len=3) -> list[Track]:
    T: dict[int, Track] = {}
    for fr in frames:
        for p in fr["persons"]:
            if p["id"] < 0:
                continue
            tr = T.setdefault(p["id"], Track(p["id"]))
            tr.ts.append(fr["t"]); tr.persons.append(p)
    return [t for t in T.values() if len(t.ts) >= min_len]


def merge_fragments(tracks: list[Track], max_gap=2.5, max_overlap=1.5, max_dist=1.2) -> list[Track]:
    """
    Склеивает треки одного человека, у которого трекер сменил ID (например, при вставании из-за стола).
    Допускается небольшое перекрытие по времени; расстояние — между последней и первой точками.
    """
    tracks = sorted(tracks, key=lambda t: t.t0)
    merged: list[Track] = []
    for tr in tracks:
        best, best_d = None, None
        for m in merged:
            gap = tr.t0 - m.t1
            if -max_overlap <= gap <= max_gap:
                a = m.centers()[-1]; b = tr.centers()[0]
                scale = max(m.height, tr.height, 1.0)
                d = float(np.hypot(*(a - b))) / scale
                if d <= max_dist and (best_d is None or d < best_d):
                    best, best_d = m, d
        if best is None:
            merged.append(Track(tr.id, list(tr.ts), list(tr.persons)))
        else:
            # При перекрытии по времени устаревшие отсчёты прежнего ID отбрасываем: актуален новый трек
            keep = [(t, p) for t, p in zip(best.ts, best.persons) if t < tr.t0]
            pairs = sorted(keep + list(zip(tr.ts, tr.persons)), key=lambda x: x[0])
            best.ts = [t for t, _ in pairs]; best.persons = [p for _, p in pairs]
    return merged


def real_tracks(tracks: list[Track], ratio=0.45) -> tuple[list[Track], list[Track]]:
    """Делит на реальных людей в помещении и «призраков» (отражения/люди за стеклом: заметно мельче главных)."""
    if not tracks:
        return [], []
    hmax = max(t.height for t in tracks)
    real = [t for t in tracks if t.height >= ratio * hmax]
    ghosts = [t for t in tracks if t.height < ratio * hmax]
    return real, ghosts


def pick_cashier(real: list[Track], workstation: Optional[list] = None) -> Optional[Track]:
    """
    Кассир — человек у рабочего места. Если известен бокс ноутбука, берём самый близкий к нему из
    достаточно долгих треков (клиент тоже может сидеть всё видео); иначе — самый долгий трек.
    """
    if not real:
        return None
    longest = max(real, key=lambda t: t.duration)
    if workstation is None:
        return longest
    lx, ly = (workstation[0] + workstation[2]) / 2, (workstation[1] + workstation[3]) / 2
    cands = [t for t in real if t.duration >= 0.25 * longest.duration]

    def dist(t):
        c = np.median(t.centers(), axis=0)
        return float(np.hypot(c[0] - lx, c[1] - ly)) / max(t.height, 1.0)
    return min(cands, key=dist)
