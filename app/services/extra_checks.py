"""
Қосымша тексерістер: трек/поза/зат детекцияларынан (Perceiver записи) және кадр статистикасынан.
Барлық арақашықтықтар кассир биіктігімен (H) өлшенеді — ракурсқа тәуелділік аз.
Әр функция таза: frames, scene -> [event dict] (бос тізім = белгі жоқ).
"""
from typing import Optional

import cv2
import numpy as np

from app.services.behavior import Scene

PHONE_CONF = 0.40
PHONE_WRIST_H = 0.30      # телефон кассирдің білегінен ≤0.3H
PHONE_MIN_HITS = 2        # кемінде 2 рет
PHONE_WINDOW = 6.0        # ... 6 секунд ішінде


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _visitors(sc: Scene, min_ratio=0.5):
    """Кассирден басқа, бөлмедегі нақты адамдар (шыны артындағы шағын «елестер» емес)."""
    return [t for t in sc.real if t is not sc.cashier and t.height >= min_ratio * sc.H]


# ── 1. Телефон кассирдің қолында ──────────────────────────────────
def _workstation_person(sc: Scene):
    """Кассир = жұмыс орнына (ноутбукке) ең жақын ұзақ трек. Клиент те видео бойы отыруы мүмкін."""
    if sc.laptop_box is None:
        return sc.cashier
    lx, ly = _center(sc.laptop_box)
    longs = [t for t in sc.real if t.duration >= 0.4 * sc.duration] or sc.real
    def dist(t):
        c = np.median(t.centers(), axis=0)
        return np.hypot(c[0] - lx, c[1] - ly) / max(t.height, 1)
    return min(longs, key=dist, default=sc.cashier)


def phone_in_hand(frames: list[dict], sc: Scene) -> list[dict]:
    c = _workstation_person(sc)
    if c is None:
        return []
    H = max(c.height, 1.0)
    hits = []
    for f in frames:
        phones = [o for o in f["objs"] if o["cls"] == "cell phone" and o["conf"] >= PHONE_CONF]
        if not phones:
            continue
        p = c.at(f["t"], 0.6)
        if p is None:
            continue
        wrists = [(x, y) for x, y, conf in (p["kpts"][9], p["kpts"][10]) if conf > 0.3]
        nose = p["kpts"][0]
        others = [q for q in f["persons"] if q is not p and q["box"] != p["box"]]
        for o in phones:
            ox, oy = _center(o["box"])
            # Телефон басқа адамның қолына жақынырақ болса — ол клиенттің телефоны
            def wd(q):
                return min([np.hypot(ox - x, oy - y) for x, y, cf in (q["kpts"][9], q["kpts"][10]) if cf > 0.3] or [1e9])
            if others and min(wd(q) for q in others) < wd(p):
                continue
            in_hand = any(np.hypot(ox - x, oy - y) <= PHONE_WRIST_H * H for x, y in wrists)
            at_face = nose[2] > 0.3 and np.hypot(ox - nose[0], oy - nose[1]) <= 0.35 * H
            # бет тұсында (сөйлесу) — кез келген сенімділікпен; тек қолда — жоғары сенімділікпен (ақша бумасымен шатаспау үшін)
            if at_face or (in_hand and o["conf"] >= 0.6):
                hits.append((f["t"], o))
                break
    for t, o in hits:
        win = [h for h in hits if t <= h[0] <= t + PHONE_WINDOW]
        if len(win) >= PHONE_MIN_HITS:
            best = max(win, key=lambda h: h[1]["conf"])
            return [{"type": "phone_use", "t": t, "evidence_t": best[0], "box": best[1]["box"],
                     "score": best[1]["conf"], "confidence": round(min(0.9, 0.4 + best[1]["conf"]), 2)}]
    return []


# ── 2. Камера жабылған / көрініс жоғалды ───────────────────────────
def frame_stats(frame_bgr: np.ndarray) -> dict:
    g = cv2.cvtColor(cv2.resize(frame_bgr, (116, 212)), cv2.COLOR_BGR2GRAY)
    return {"mean": float(g.mean()), "std": float(g.std()),
            "sharp": float(cv2.Laplacian(g, cv2.CV_64F).var())}


def camera_blocked(stats: list[tuple[float, dict]], min_sec=2.0) -> list[dict]:
    """Кадр біркелкі (жабылған), өте қараңғы немесе өте ашық болып ≥ min_sec тұрса."""
    bad = [(t, s) for t, s in stats if s["std"] < 10 or s["mean"] < 18 or s["mean"] > 240]
    if not bad:
        return []
    run_start, prev = None, None
    for t, s in bad:
        if run_start is None or t - prev > 1.2:
            run_start = t
        prev = t
        if prev - run_start >= min_sec:
            return [{"type": "camera_blocked", "t": run_start, "evidence_t": run_start + 0.5, "confidence": 0.85,
                     "info": {"mean": round(s["mean"], 1), "std": round(s["std"], 1)}}]
    return []


# ── 3. Кассир клиентті қалдырып кетті ─────────────────────────────
def left_with_client(sc: Scene) -> list[dict]:
    for t0, t1 in sc.away:
        tv = sc.vacant_from.get((t0, t1), t0)
        for v in _visitors(sc):
            # клиент кассир кеткенге дейін ≥3 с бөлмеде болып, кеткен сәтте де бар
            if v.t0 <= tv - 3.0 and v.t1 >= tv + 2.0:
                return [{"type": "left_with_client", "t": tv, "evidence_t": tv + 1.0, "confidence": 0.75}]
    return []


# ── 4. Кассир жоқта бөлмеге адам кірді ────────────────────────────
def entered_while_away(sc: Scene) -> list[dict]:
    for t0, t1 in sc.away:
        tv = sc.vacant_from.get((t0, t1), t0)
        for v in _visitors(sc):
            if tv + 1.0 <= v.t0 <= t1 and v.duration >= 2.0:
                return [{"type": "entered_while_away", "t": v.t0, "evidence_t": v.t0 + 1.0, "confidence": 0.7}]
    return []


# ── 5. Бөтен адам кассирдің жанында (оның жағында) ────────────────
def outsider_near_cashier(sc: Scene, dist_h=0.30, min_sec=2.0) -> list[dict]:
    c = sc.cashier
    if c is None:
        return []
    for v in _visitors(sc):
        # Камераның алдынан өтіп бара жатқан адам (кассирден әлдеқайда үлкен) кассир жағында емес
        if not (0.6 <= v.height / max(c.height, 1.0) <= 1.6):
            continue
        close = []
        for t, p in zip(v.ts, v.persons):
            q = c.at(t, 0.8)
            if q is None:
                continue
            d = np.hypot(*(np.array(_center(p["box"])) - np.array(_center(q["box"])))) / sc.H
            # Қатты қабаттасқан бокс — сол адамның қайталанған детекциясы, бөтен адам емес
            if d <= dist_h and _iou(p["box"], q["box"]) < 0.3:
                close.append(t)
        if close:
            run0, prev = close[0], close[0]
            for t in close[1:]:
                if t - prev > 1.2:
                    run0 = t
                prev = t
                if prev - run0 >= min_sec:
                    return [{"type": "outsider_near_cashier", "t": run0, "evidence_t": run0 + 1.0, "confidence": 0.7}]
    return []


# ── 6. Камера бұрылды / жылжытылды ─────────────────────────────────
def small_gray(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame_bgr, (116, 212)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g


def camera_moved(grays: list[tuple[float, np.ndarray]], min_sec=2.0, min_shift=0.05) -> list[dict]:
    """Фон бастапқы кадрға қарағанда тұтас ығысса (≥5% кадр) және ≥2 с сол күйде тұрса — камера қозғалған."""
    if len(grays) < 6:
        return []
    ref = np.median(np.stack([g for _, g in grays[:3]]), axis=0).astype(np.float32)
    H, W = ref.shape
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    moved = []
    for t, g in grays[3:]:
        (dx, dy), resp = cv2.phaseCorrelate(ref, g, win)
        if resp > 0.08 and np.hypot(dx / W, dy / H) >= min_shift:
            moved.append((t, dx / W, dy / H))
    if not moved:
        return []
    run0, prev = moved[0][0], moved[0][0]
    for t, dx, dy in moved[1:]:
        if t - prev > 1.2:
            run0 = t
        prev = t
        if prev - run0 >= min_sec:
            return [{"type": "camera_moved", "t": run0, "evidence_t": run0 + 0.5, "confidence": 0.8,
                     "info": {"shift": round(float(np.hypot(dx, dy)) * 100, 1)}}]
    return []
