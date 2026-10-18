"""
Конвейер анализа видео кассовой дисциплины (полностью локальный).

1. Люди: YOLOv8-pose + ByteTrack на ~2 кадрах/с (шаг подбирается под остаток времени).
2. Сцена: кассир, место за столом, интервалы «кассир отошёл» (behavior.analyze_scene).
3. Состояния: яркость экрана ноутбука (ПК заблокирован или нет).
4. Точечные проверки OWLv2: предметы на столе и у кассира (probes.plan_probes).
5. События -> нарушения: тип, таймкод, описание, кадры-доказательства, норма.

Жёсткий дедлайн (5 минут с момента получения видео) соблюдается на каждом этапе.
"""
import base64
import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import MAX_PROCESSING_TIME
from app.services.behavior import analyze_scene, Scene
from app.services.state_checks import laptop_brightness, pc_left_unlocked
from app.services.door import prep as door_prep, door_timeline, door_left_open
from app.services import extra_checks as XC
from app.services.probes import plan_probes
from app.services.rule_mapper import RuleMapper

logger = logging.getLogger(__name__)

SAMPLE_FPS = 2.0            # целевая частота анализа
SAFETY_SEC = 20.0           # запас до дедлайна на сборку ответа
OBJECT_EVERY = 3            # детектор предметов (ноутбук, телефон) — на каждом 3-м анализируемом кадре

# Пороги OWLv2, подобранные по обучающему датасету (tools/eval_pipeline.py)
OWL_TH = {
    "valuables": {"a bundle of banknotes": 0.20, "a tablet computer": 0.40, "a green bag": 0.25, "a handbag": 0.30},
    "personal": {"a handbag": 0.40, "a jacket or coat": 0.35},
    "drink_food": {"a cup": 0.30, "food or a snack": 0.30, "a bottle": 0.35},
}

TYPE_NAMES = {
    "door_open": "Кассалық бөлменің есігі ашық қалдырылған",
    "phone_use": "Кассир ұялы телефон қолданып отыр",
    "camera_blocked": "Бейнекамера жабылған немесе көрініс жоқ",
    "camera_moved": "Бейнекамера бұрылған немесе жылжытылған",
    "left_with_client": "Кассир клиентті қалдырып жұмыс орнынан кетті",
    "entered_while_away": "Кассир жоқта бөлмеге бөтен адам кірді",
    "outsider_near_cashier": "Бөтен адам кассирдің жұмыс аймағына өтті",
    "pc_unlocked": "Кассир кеткенде компьютер блокталмаған",
    "valuables_unattended": "Құндылықтар үстелде қараусыз қалдырылған",
    "personal_items": "Жеке заттар кассалық аймақта",
    "food_drink": "Жұмыс орнында тамақ/сусын",
}

_singletons_lock = threading.Lock()
_perceiver = None
_owl = None


def get_perceiver():
    global _perceiver
    with _singletons_lock:
        if _perceiver is None:
            from app.services.perception import Perceiver
            _perceiver = Perceiver()
        return _perceiver


def get_owl():
    global _owl
    with _singletons_lock:
        if _owl is None:
            from app.services.owl import OwlScorer
            _owl = OwlScorer()
        return _owl


def warm_up() -> None:
    """Загружает модели заранее, чтобы первое видео не платило за загрузку."""
    try:
        get_perceiver(); get_owl()
        logger.info("Модели загружены")
    except Exception:
        logger.exception("Не удалось заранее загрузить модели")


def _fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}" if m else f"{s:.1f} с"


def build_item_events(scene: Scene, probe_results: list[dict], frames: Optional[list[dict]] = None) -> list[dict]:
    """
    Чистая функция: результаты OWL-проб -> события. Используется и в живом режиме, и в оффлайн-оценке по кэшу.
    probe_results: [{"t", "kind", "roi", "scores": {запрос: [уверенность, бокс в координатах кропа]}}]
    """
    kind_map = {
        "desk_away": [("valuables_unattended", "valuables")],
        "desk_seated": [("personal_items", "personal"), ("food_drink", "drink_food")],
        "person": [("personal_items", "personal"), ("food_drink", "drink_food")],
    }
    def carried_by_person(t, full_box) -> bool:
        """Предмет в руках/на человеке (центр бокса внутри бокса человека) — не «оставлен на столе»."""
        if not frames:
            return False
        rec = min(frames, key=lambda f: abs(f["t"] - t))
        cx, cy = (full_box[0] + full_box[2]) / 2, (full_box[1] + full_box[3]) / 2
        return any(p["box"][3] - p["box"][1] > 60 and p["box"][0] <= cx <= p["box"][2] and p["box"][1] <= cy <= p["box"][3]
                   for p in rec["persons"])

    def on_laptop(full_box) -> bool:
        """Бокс «планшета» внутри уже известного ноутбука — это сам ноутбук, а не оставленная вещь."""
        lb = scene.laptop_box
        if not lb:
            return False
        pad_x, pad_y = 0.25 * (lb[2] - lb[0]), 0.25 * (lb[3] - lb[1])
        cx, cy = (full_box[0] + full_box[2]) / 2, (full_box[1] + full_box[3]) / 2
        return lb[0] - pad_x <= cx <= lb[2] + pad_x and lb[1] - pad_y <= cy <= lb[3] + pad_y

    # «Оставлено» = предмет остаётся на столе до конца интервала отсутствия: учитываем только последнюю
    # пробу каждого интервала (иначе предмет, который кассир как раз убирает в сейф, даст ложное срабатывание)
    last_away_probe = {}
    for pr in probe_results:
        if pr["kind"] == "desk_away":
            iv = next(((a, b) for a, b in scene.away if a <= pr["t"] <= b), None)
            if iv is not None and (iv not in last_away_probe or pr["t"] > last_away_probe[iv]["t"]):
                last_away_probe[iv] = pr
    last_ids = {id(p) for p in last_away_probe.values()}

    found: dict[str, dict] = {}
    for pr in probe_results:
        if pr["kind"] == "desk_away" and id(pr) not in last_ids:
            continue
        x1, y1 = max(0, pr["roi"][0]), max(0, pr["roi"][1])
        for etype, group in kind_map.get(pr["kind"], []):
            for q, th in OWL_TH[group].items():
                if q in pr["scores"] and pr["scores"][q][0] >= th:
                    sc, bx = pr["scores"][q]
                    full = [bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1]
                    if on_laptop(full):
                        continue
                    if pr["kind"] == "desk_away" and carried_by_person(pr["t"], full):
                        continue
                    # Зелёная сумка — инкассаторская (рабочая), а не личная сумка работника
                    if etype == "personal_items" and q == "a handbag" and pr["scores"].get("a green bag", [0])[0] >= sc:
                        continue
                    cur = found.get(etype)
                    if cur is None or sc > cur["score"]:
                        found[etype] = {"type": etype, "t": pr["t"], "score": sc, "query": q, "box": full,
                                        "probe_kind": pr["kind"], "box_kind": "item"}
    events = []
    for etype, f in found.items():
        if etype == "valuables_unattended":
            t_leave = next((a for a, b in scene.away if a <= f["t"] <= b), f["t"])
            f = {**f, "t": t_leave, "evidence_t": f["t"]}
        f["confidence"] = round(min(0.9, 0.45 + f["score"]), 2)
        events.append(f)
    return events


class AnalysisPipeline:
    def __init__(self):
        self.rules = RuleMapper()

    # ───────────────────────── публичный вход ─────────────────────────
    def analyze(self, file_path: str, deadline: Optional[float] = None,
                progress: Optional[Callable[[float, str], None]] = None) -> dict:
        t_start = time.time()
        deadline = deadline or (t_start + MAX_PROCESSING_TIME)
        progress = progress or (lambda p, s: None)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видео")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = n_frames / fps if fps > 0 else 0.0

        progress(3, "Модельдер жүктелуде")
        perceiver = get_perceiver()
        perceiver.reset_tracking()

        # ── 1. Люди (поза + трекинг) ────────────────────────────
        frames, lap_series = self._perceive(cap, perceiver, fps, n_frames, duration, deadline, progress)
        # ── 2. Сцена ────────────────────────────────────────────
        progress(55, "Көрініс талдануда")
        scene = analyze_scene(frames, duration, w, h)
        # ── 3–4. События ───────────────────────────────────────
        events = []
        events += self._extra_events(frames, scene)
        events += self._door_events(frames)
        events += self._pc_events(scene, lap_series)
        progress(60, "Заттар тексерілуде")
        events += self._item_events(cap, fps, scene, deadline, progress, frames)
        # ── 5. Нарушения с доказательствами ─────────────────────
        progress(92, "Дәлелдер дайындалуда")
        violations = self._build_violations(cap, fps, frames, events)
        cap.release()

        verdict = "violation" if violations else "no_violation"
        elapsed = time.time() - t_start
        logger.info("Анализ завершён за %.1f с | вердикт: %s | нарушений: %d", elapsed, verdict, len(violations))
        progress(100, "Дайын")
        return {
            "verdict": verdict,
            "violations": violations,
            "video_info": {
                "resolution": f"{w}x{h}", "fps": round(fps, 2), "frame_count": n_frames,
                "duration_sec": round(duration, 2), "processing_sec": round(elapsed, 1),
            },
        }

    # ───────────────────────── этап 1 ─────────────────────────
    def _perceive(self, cap, perceiver, fps, n_frames, duration, deadline, progress):
        """Читает видео последовательно, анализируя каждый step-й кадр; шаг растёт, если не хватает времени."""
        base_step = max(1, int(round(fps / SAMPLE_FPS)))
        step = base_step
        frames: list[dict] = []
        self._grays: list = []
        self._stats: list = []
        self._sgrays: list = []
        lap_series: list[tuple[float, float]] = []
        lap_box: Optional[list] = None
        idx = 0
        t_loop = time.time()
        # Этапу восприятия отводим часть остатка; остальное — на OWL-пробы и сборку ответа
        budget_end = t_loop + max(20.0, (deadline - t_loop - SAFETY_SEC) * 0.6)
        analysed = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    rec = perceiver.process(frame, idx, fps, with_objects=(analysed % OBJECT_EVERY == 0))
                    frames.append(rec)
                    self._grays.append(door_prep(frame))
                    self._stats.append((rec["t"], XC.frame_stats(frame)))
                    self._sgrays.append((rec["t"], XC.small_gray(frame)))
                    for o in rec["objs"]:
                        if o["cls"] == "laptop" and o["conf"] >= 0.5:
                            lap_box = o["box"]
                    if lap_box is not None:
                        v = laptop_brightness(frame, lap_box)
                        if v is not None:
                            lap_series.append((rec["t"], v))
                    analysed += 1
                    if analysed % 5 == 0:
                        progress(5 + 48 * min(1.0, idx / max(1, n_frames)), "Адамдар анықталуда")
                    if analysed % 8 == 0:
                        # Регулятор темпа: если не успеваем — прореживаем кадры
                        now = time.time()
                        per_frame = (now - t_loop) / analysed
                        remaining = max(0.0, (n_frames - idx) / step)
                        projected = remaining * per_frame
                        allowed = max(1.0, budget_end - now)
                        if projected > allowed:
                            step = min(base_step * 8, int(np.ceil(step * projected / allowed)))
                            logger.info("Шаг кадров увеличен до %d (прогноз %.0f с > доступно %.0f с)", step, projected, allowed)
            idx += 1
            if time.time() > deadline - SAFETY_SEC - 25:
                logger.warning("Дедлайн близко: остановка этапа 1 на кадре %d/%d", idx, n_frames)
                break
        # Текущий видео без людей/предметов — всё равно возвращаем пустые записи
        return frames, lap_series

    # ───────────────────────── этап 3 ─────────────────────────
    def _extra_events(self, frames, scene) -> list[dict]:
        """Телефон, камера, адамдардың орналасуы — әр тексеріс өз алдына, біреуінің қатесі басқасын тоқтатпайды."""
        out = []
        checks = (
            lambda: XC.phone_in_hand(frames, scene),
            lambda: XC.camera_blocked(self._stats),
            lambda: XC.camera_moved(self._sgrays),
            lambda: XC.left_with_client(scene),
            lambda: XC.entered_while_away(scene),
            lambda: XC.outsider_near_cashier(scene),
        )
        for chk in checks:
            try:
                out += chk()
            except Exception:
                logger.exception("Қосымша тексеріс орындалмады")
        self._stats, self._sgrays = [], []
        return out

    def _door_events(self, frames) -> list[dict]:
        try:
            r = door_left_open(door_timeline(self._grays, frames))
        except Exception:
            logger.exception("Проверка двери не выполнена")
            return []
        finally:
            self._grays = []
        if not r:
            return []
        return [{"type": "door_open", "t": r["t"], "evidence_t": r["evidence_t"], "confidence": 0.8,
                 "box": r["box"], "info": r, "box_kind": "door"}]

    def _pc_events(self, scene: Scene, lap_series) -> list[dict]:
        out = []
        for t0, t1 in scene.away:
            r = pc_left_unlocked(lap_series, t0)
            if r:
                out.append({"type": "pc_unlocked", "t": t0, "confidence": 0.7, "info": r,
                            "box_kind": "laptop"})
                break                      # одного события достаточно
        return out

    # ───────────────────────── этап 4 ─────────────────────────
    def _item_events(self, cap, fps, scene: Scene, deadline, progress, frames=None) -> list[dict]:
        probes = plan_probes(scene)
        if not probes:
            return []
        owl = get_owl()
        results = []
        for i, pr in enumerate(probes):
            if time.time() > deadline - SAFETY_SEC - 6:
                logger.warning("Дедлайн: OWL-пробы остановлены (%d/%d)", i, len(probes))
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pr.t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = pr.roi
            crop = fr[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            res = owl.score(crop, pr.queries)
            results.append({"t": pr.t, "kind": pr.kind, "roi": pr.roi,
                            "scores": {q: [s, b] for q, (s, b) in res.items()}})
            progress(60 + 30 * (i + 1) / len(probes), "Заттар тексерілуде")
        return build_item_events(scene, results, frames)

    # ───────────────────────── этап 5 ─────────────────────────
    def _build_violations(self, cap, fps, frames, events) -> list[dict]:
        violations = []
        for ev in sorted(events, key=lambda e: e["t"]):
            ev_t = ev.get("evidence_t", ev["t"])
            ev_frames = self._evidence(cap, fps, frames, ev, ev_t)
            if not ev_frames:
                continue
            rule = self.rules.rule(ev["type"])
            violations.append({
                "type": TYPE_NAMES[ev["type"]],
                "timecode_sec": round(float(ev["t"]), 1),
                "description": self._describe(ev),
                "confidence": ev["confidence"],
                "evidence_frames": ev_frames,
                "rule": rule,
                "regulatory_reference": self.rules.format_reference(ev["type"]),
                "limitations": self.rules.format_limitations(ev["type"]),
            })
        return violations

    def _evidence(self, cap, fps, frames, ev, ev_t) -> list[dict]:
        out = []
        offsets = (-1.5, 0.0, 1.5) if ev["type"] == "pc_unlocked" else (0.0, 1.5)
        if ev["type"] in ("door_open", "camera_moved", "entered_while_away", "outsider_near_cashier", "left_with_client"):
            offsets = (0.0, 2.0)
        if ev["type"] == "camera_blocked":
            offsets = (-2.0, 0.0, 1.0)
        for off in offsets:
            t = max(0.0, ev_t + off)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            rec = min(frames, key=lambda f: abs(f["t"] - t)) if frames else None
            if rec:
                for p in rec["persons"]:
                    if p["box"][3] - p["box"][1] > 60:
                        x1, y1, x2, y2 = [int(v) for v in p["box"]]
                        cv2.rectangle(fr, (x1, y1), (x2, y2), (60, 200, 60), 2)
            if ev.get("box") is not None and abs(t - ev_t) < 0.1:
                x1, y1, x2, y2 = [int(v) for v in ev["box"]]
                cv2.rectangle(fr, (x1, y1), (x2, y2), (40, 40, 255), 3)
            out.append({"timestamp_sec": round(t, 1), "image_base64": self._jpeg_b64(fr)})
        return out[:3]

    @staticmethod
    def _jpeg_b64(frame: np.ndarray, quality: int = 80, max_dim: int = 1280) -> str:
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            r = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _describe(ev: dict) -> str:
        t = _fmt_time(ev["t"])
        etype = ev["type"]
        if etype == "phone_use":
            return (f"{t} мезетінде кассирдің қолында немесе бет тұсында ұялы телефон көрінеді (анықтау сенімділігі "
                    f"{ev.get('score', 0):.2f}). Фото/бейне түсіру функциясы бар байланыс құралын кассалық торапқа "
                    "кіргізуге болмайды.")
        if etype == "camera_blocked":
            return (f"{t} мезетінен бастап кадр бірнеше секунд бойы біркелкі/қараңғы: камера көрінісі жабылған "
                    "немесе жоғалған. Кассалық торап бейнебақылау камерасымен жабдықталуы тиіс.")
        if etype == "camera_moved":
            return (f"{t} мезетінен бастап бүкіл көрініс бастапқы кадрға қарағанда ығысқан "
                    f"(шамамен {ev.get('info', {}).get('shift', '?')}% кадр): камера бұрылған немесе жылжытылған болуы мүмкін.")
        if etype == "left_with_client":
            return (f"{t} мезетінде кассир жұмыс орнын тастап кетті, ал клиент бөлмеде қалды. Клиентке қызмет "
                    "көрсету аяқталғанға дейін жұмыс орнынан кетуге болмайды.")
        if etype == "entered_while_away":
            return (f"{t} мезетінде кассир жұмыс орнында жоқ кезде бөлмеге басқа адам кірді. "
                    "Кассалық торап оқшауланған болуы тиіс.")
        if etype == "outsider_near_cashier":
            return (f"{t} мезетінен бастап басқа адам кассирдің дәл жанында, оның жұмыс аймағында тұр "
                    "(бірнеше секунд бойы). Бөтен адамның кассир жағына өтуі кассалық тораптың оқшаулануын бұзады.")
        if etype == "door_open":
            return (f"{t} мезетінен бастап кассалық бөлменің есігі ашық күйде қалды: адам өткен соң есік жапырағы "
                    "бұрынғы (жабық) орнына оралмады және бірнеше секунд бойы ашық тұр. Кассалық торап бір жабылатын "
                    "кіреберіспен оқшауланған болуы тиіс.")
        if etype == "pc_unlocked":
            return (f"{t} мезетінде кассир жұмыс орнынан кетті, ал ноутбук экраны жарық күйінде қалды "
                    f"(жарықтығы {ev['info']['brightness']}, кеткенге дейін құлыптау белгісі байқалмады). "
                    "Бұл компьютер блокталмай қалуы мүмкін екенін көрсетеді.")
        if etype == "valuables_unattended":
            what = {"a bundle of banknotes": "банкноттар", "a tablet computer": "планшет",
                    "a green bag": "сөмке", "a handbag": "сөмке"}.get(ev.get("query"), "зат")
            return (f"{t} мезетінде кассир жұмыс орнынан кетті, ал үстелде {what} қалғаны көрінеді "
                    f"(анықтау сенімділігі {ev['score']:.2f}). Оны жабылатын сақтау орнына салу байқалмады.")
        if etype == "personal_items":
            what = {"a handbag": "сөмке", "a jacket or coat": "сырт киім"}.get(ev.get("query"), "жеке зат")
            return (f"{t} мезетінде кассирдің жұмыс аймағында {what} көрінеді (сенімділік {ev['score']:.2f}). "
                    "Жеке заттар кассалық торапта сақталмауы тиіс.")
        if etype == "food_drink":
            what = {"a cup": "шыны-аяқ/кесе", "food or a snack": "тамақ", "a bottle": "бөтелке"}.get(ev.get("query"), "тағам/сусын")
            return (f"{t} мезетінде кассирдің жұмыс орнында {what} көрінеді (сенімділік {ev['score']:.2f}). "
                    "Бұл кассалық аймақтағы жеке зат ретінде қаралуы мүмкін.")
        return "Ықтимал бұзушылық белгісі анықталды."
