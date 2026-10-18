"""
Тесты сервисов обработки видео.

Проверяют:
  - Чтение видео через OpenCV
  - Детекцию объектов YOLO
  - Классификацию событий
  - Извлечение доказательных кадров
  - Сопоставление с нормативными источниками
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.video_reader import VideoReader
from app.services.evidence_extractor import EvidenceExtractor
from app.services.rule_mapper import RuleMapper


# ═══════════════════════════════════════════════════
# Утилиты
# ═══════════════════════════════════════════════════

def make_test_video(path: str, duration_sec: float = 2.0, fps: int = 10):
    """Создаёт короткое тестовое видео."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 240
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    frames = int(duration_sec * fps)
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)
        cv2.rectangle(frame, (100, 80), (220, 200), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def test_video(tmp_path):
    path = str(tmp_path / "test.mp4")
    make_test_video(path, duration_sec=1.0, fps=10)
    return path


# ═══════════════════════════════════════════════════
# Тесты VideoReader
# ═══════════════════════════════════════════════════

class TestVideoReader:
    def test_open_video(self, test_video):
        reader = VideoReader(test_video)
        assert reader.cap is not None
        assert reader.cap.isOpened()
        reader.release()

    def test_get_info(self, test_video):
        reader = VideoReader(test_video)
        info = reader.get_info()
        assert info.width == 320
        assert info.height == 240
        assert info.fps > 0
        assert info.frame_count > 0
        reader.release()

    def test_read_frames(self, test_video):
        reader = VideoReader(test_video)
        count = 0
        while True:
            data = reader.read_frame()
            if data is None:
                break
            count += 1
        assert count > 0
        reader.release()

    def test_seek(self, test_video):
        reader = VideoReader(test_video)
        frame = reader.seek_frame(0)
        assert frame is not None
        assert frame.shape[0] == 240
        assert frame.shape[1] == 320
        reader.release()

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoReader(str(tmp_path / "nonexistent.mp4"))


# ═══════════════════════════════════════════════════
# Тесты EvidenceExtractor
# ═══════════════════════════════════════════════════

class TestEvidenceExtractor:
    def test_extract_frame(self, test_video):
        reader = VideoReader(test_video)
        extractor = EvidenceExtractor(reader)
        frames = extractor.extract(timestamp_sec=0.0, fps=10, max_frames=1)
        assert len(frames) >= 1
        assert "image_base64" in frames[0]
        assert "timestamp_sec" in frames[0]
        assert len(frames[0]["image_base64"]) > 100
        reader.release()

    def test_extract_multiple_frames(self, test_video):
        reader = VideoReader(test_video)
        extractor = EvidenceExtractor(reader)
        frames = extractor.extract(timestamp_sec=0.0, fps=10, max_frames=3)
        assert len(frames) <= 3
        reader.release()

    def test_base64_validity(self, test_video):
        import base64
        reader = VideoReader(test_video)
        extractor = EvidenceExtractor(reader)
        frames = extractor.extract(timestamp_sec=0.0, fps=10, max_frames=1)
        if frames:
            decoded = base64.b64decode(frames[0]["image_base64"])
            assert len(decoded) > 0
            # Проверяем сигнатуру JPEG
            assert decoded[:2] == b"\xff\xd8"
        reader.release()


# ═══════════════════════════════════════════════════
# Тесты RuleMapper
# ═══════════════════════════════════════════════════

class TestRuleMapper:
    def test_every_rule_is_complete(self):
        from app.services.rule_mapper import RULES
        assert RULES
        for code, r in RULES.items():
            assert r["document"].startswith("Постановление Правления") and "РК" in r["document"], code
            assert r["clause"].startswith("п."), code
            assert r["quote"].strip(), code
            assert r["url"].startswith("https://adilet.zan.kz/"), code
            assert r["limitation"].strip(), code

    def test_quotes_exist_in_source_text(self):
        """Цитаты нормы должны дословно присутствовать в тексте документов (защита от выдуманных ссылок)."""
        import re
        from pathlib import Path
        from app.services.rule_mapper import RULES, DOC_231, DOC_14, DOC_53
        base = Path(__file__).resolve().parent.parent / "docs" / "normative"
        if not (base / "p231.txt").exists():
            pytest.skip("Тексты нормативных документов не найдены")
        norm = lambda t: re.sub(r"\s+", " ", t)
        src = {DOC_231: norm((base / "p231.txt").read_text()), DOC_14: norm((base / "p14.txt").read_text()),
               DOC_53: norm((base / "p53.txt").read_text())}
        for code, r in RULES.items():
            assert norm(r["quote"]) in src[r["document"]], f"Цитата для {code} не найдена в источнике"

    def test_unknown_event_type(self):
        mapper = RuleMapper()
        rule = mapper.map("unknown_type")
        assert rule["document"] is None
        assert rule["verified"] is False
        assert mapper.rule("unknown_type") is None
        assert mapper.format_reference("unknown_type") is None

    def test_reference_and_limitations(self):
        mapper = RuleMapper()
        assert "adilet.zan.kz" in mapper.format_reference("valuables_unattended")
        assert mapper.format_limitations("valuables_unattended")
        r = mapper.rule("valuables_unattended")
        assert set(r) == {"document", "clause", "quote", "note", "url"}

    def test_every_pipeline_event_type_has_a_rule(self):
        """Каждый тип, который может выдать пайплайн, обязан иметь норму (иначе в ответе rule=null)."""
        from app.services.analysis_pipeline import TYPE_NAMES
        from app.services.rule_mapper import RULES
        assert set(TYPE_NAMES) <= set(RULES), set(TYPE_NAMES) - set(RULES)


# ═══════════════════════════════════════════════════
# Тесты поведенческого анализа (чистые функции)
# ═══════════════════════════════════════════════════

def _person(pid, box, kpts=None):
    return {"id": pid, "box": box, "conf": 0.9, "kpts": kpts or [[0, 0, 0.0]] * 17}


def _frames(spec, step=0.5):
    """spec: список (t, [persons]) -> записи кадров."""
    return [{"t": t, "idx": int(t * 30), "persons": ps, "objs": []} for t, ps in spec]


class TestBehavior:
    def test_seated_then_standing_creates_away_interval(self):
        from app.services.behavior import analyze_scene
        spec = []
        for i in range(30):                      # 0–15 с: сидит (бокс низкий)
            spec.append((i * 0.5, [_person(1, [100, 300, 200, 400])]))
        for i in range(30, 60):                  # 15–30 с: стоит и уходит (бокс высокий)
            spec.append((i * 0.5, [_person(1, [300, 100, 380, 400])]))
        # последние 8 с кассира нет в зоне стола (вышла)
        spec = spec[:44] + [(i * 0.5, []) for i in range(44, 60)]
        sc = analyze_scene(_frames(spec), 30.0, 464, 848)
        assert sc.cashier is not None
        assert sc.away and sc.away[0][0] <= 16.0 and sc.away[0][1] >= 29.0
        assert 15.0 <= sc.vacant_from[sc.away[0]] <= 30.0

    def test_standing_at_the_desk_is_not_leaving(self):
        from app.services.behavior import analyze_scene
        spec = [(i * 0.5, [_person(1, [100, 300, 200, 400])]) for i in range(30)]
        spec += [(i * 0.5, [_person(1, [110, 100, 190, 400])]) for i in range(30, 60)]   # встала, но осталась у стола
        sc = analyze_scene(_frames(spec), 30.0, 464, 848)
        assert sc.away == []

    def test_always_seated_has_no_away(self):
        from app.services.behavior import analyze_scene
        spec = [(i * 0.5, [_person(1, [100, 300, 200, 420])]) for i in range(40)]
        sc = analyze_scene(_frames(spec), 20.0, 464, 848)
        assert sc.away == []

    def test_small_background_people_are_ghosts(self):
        from app.services.tracks import build_tracks, real_tracks
        spec = [(i * 0.5, [_person(1, [100, 300, 200, 500]), _person(2, [20, 200, 50, 270])]) for i in range(20)]
        real, ghosts = real_tracks(build_tracks(_frames(spec)))
        assert [t.id for t in real] == [1] and [t.id for t in ghosts] == [2]


class TestStateChecks:
    def test_pc_left_unlocked_when_screen_stays_bright(self):
        from app.services.state_checks import pc_left_unlocked
        series = [(t, 140.0) for t in range(0, 12)]
        assert pc_left_unlocked(series, 8.0) is not None

    def test_pc_locked_before_leaving_is_not_a_violation(self):
        from app.services.state_checks import pc_left_unlocked
        series = [(t, 140.0) for t in range(0, 5)] + [(t, 105.0) for t in range(5, 12)]
        assert pc_left_unlocked(series, 8.0) is None

    def test_dark_screen_is_not_a_violation(self):
        from app.services.state_checks import pc_left_unlocked
        assert pc_left_unlocked([(t, 90.0) for t in range(12)], 8.0) is None


class TestItemEvents:
    def _scene(self):
        from app.services.behavior import Scene
        return Scene(duration=30.0, cashier=None, away=[(10.0, 25.0)])

    def test_valuables_event_anchored_at_leave_time(self):
        from app.services.analysis_pipeline import build_item_events
        probes = [{"t": 14.0, "kind": "desk_away", "roi": [0, 0, 300, 300],
                   "scores": {"a bundle of banknotes": [0.3, [10, 10, 40, 30]], "a tablet computer": [0.05, [0, 0, 1, 1]]}}]
        ev = build_item_events(self._scene(), probes)
        assert [e["type"] for e in ev] == ["valuables_unattended"]
        assert ev[0]["t"] == 10.0 and ev[0]["evidence_t"] == 14.0

    def test_item_carried_by_person_is_ignored(self):
        from app.services.analysis_pipeline import build_item_events
        frames = [{"t": 14.0, "idx": 0, "persons": [_person(1, [0, 0, 100, 200])], "objs": []}]
        probes = [{"t": 14.0, "kind": "desk_away", "roi": [0, 0, 300, 300],
                   "scores": {"a handbag": [0.6, [20, 20, 60, 60]]}}]
        assert build_item_events(self._scene(), probes, frames) == []

    def test_item_that_disappears_before_the_end_is_not_a_violation(self):
        from app.services.analysis_pipeline import build_item_events
        probes = [
            {"t": 12.0, "kind": "desk_away", "roi": [0, 0, 300, 300], "scores": {"a bundle of banknotes": [0.4, [10, 10, 40, 30]]}},
            {"t": 24.0, "kind": "desk_away", "roi": [0, 0, 300, 300], "scores": {"a bundle of banknotes": [0.05, [0, 0, 1, 1]]}},
        ]
        assert build_item_events(self._scene(), probes) == []

    def test_item_inside_laptop_box_is_ignored(self):
        from app.services.analysis_pipeline import build_item_events
        sc = self._scene(); sc.laptop_box = [100, 100, 200, 180]
        probes = [{"t": 14.0, "kind": "desk_away", "roi": [0, 0, 300, 300],
                   "scores": {"a tablet computer": [0.6, [120, 110, 180, 170]]}}]
        assert build_item_events(sc, probes) == []

    def test_green_cash_bag_is_not_personal_item(self):
        from app.services.analysis_pipeline import build_item_events
        probes = [{"t": 5.0, "kind": "desk_seated", "roi": [0, 0, 300, 300],
                   "scores": {"a handbag": [0.5, [10, 10, 40, 30]], "a green bag": [0.6, [10, 10, 40, 30]]}}]
        assert build_item_events(self._scene(), probes) == []

    def test_below_threshold_is_ignored(self):
        from app.services.analysis_pipeline import build_item_events
        probes = [{"t": 14.0, "kind": "desk_away", "roi": [0, 0, 300, 300],
                   "scores": {"a tablet computer": [0.2, [10, 10, 40, 30]]}}]
        assert build_item_events(self._scene(), probes) == []


class TestDoor:
    """Синтетика: фон неподвижен, «дверное полотно» — высокий прямоугольник в верхней части кадра."""

    def _run(self, change_from, change_to, global_change=False, n=30):
        from app.services.door import door_timeline, door_left_open
        H, W = 424, 232
        grays, frames = [], []
        for i in range(n):
            t = i * 0.5
            g = np.full((H, W), 120, np.uint8)
            if change_from <= t < change_to:
                g[20:200, 10:80] = 30          # дверь в другом положении
            if global_change and t >= 5:
                g[:] = 40                      # погас свет во всей комнате
            grays.append(g)
            frames.append({"t": t, "idx": i, "persons": [], "objs": []})
        return door_left_open(door_timeline(grays, frames))

    def test_door_left_open_is_detected(self):
        r = self._run(5.0, 99.0)
        assert r is not None and 4.5 <= r["t"] <= 6.0

    def test_door_closed_again_is_not_a_violation(self):
        assert self._run(5.0, 7.0) is None

    def test_global_lighting_change_is_ignored(self):
        assert self._run(99.0, 99.0, global_change=True) is None


class TestExtraChecks:
    """Әр жаңа тексеріс: оң жағдайда іске қосылады, қалыпты жағдайда үндемейді."""

    @staticmethod
    def _kp(wrist=None, nose=None):
        k = [[0, 0, 0.0]] * 17
        k = [list(x) for x in k]
        if wrist: k[9] = [wrist[0], wrist[1], 0.9]
        if nose: k[0] = [nose[0], nose[1], 0.9]
        return k

    def _scene(self, frames, dur=20.0):
        from app.services.behavior import analyze_scene
        return analyze_scene(frames, dur, 464, 848)

    def test_phone_at_face_detected(self):
        from app.services.extra_checks import phone_in_hand
        fr = []
        for i in range(40):
            t = i * 0.5
            fr.append({"t": t, "idx": i, "persons": [_person(1, [100, 200, 200, 420], self._kp((150, 300), (150, 230)))],
                       "objs": [{"cls": "cell phone", "conf": 0.5, "box": [140, 220, 160, 245]}] if 10 <= t <= 14 else []})
        assert [e["type"] for e in phone_in_hand(fr, self._scene(fr))] == ["phone_use"]

    def test_phone_held_by_other_person_is_ignored(self):
        from app.services.extra_checks import phone_in_hand
        fr = []
        for i in range(40):
            t = i * 0.5
            cashier = _person(1, [100, 200, 200, 420], self._kp((150, 300), (150, 230)))
            client = _person(2, [260, 200, 360, 420], self._kp((300, 300), (300, 230)))
            fr.append({"t": t, "idx": i, "persons": [cashier, client],
                       "objs": [{"cls": "cell phone", "conf": 0.8, "box": [290, 290, 310, 310]}] if 10 <= t <= 14 else []})
        assert phone_in_hand(fr, self._scene(fr)) == []

    def test_camera_blocked(self):
        from app.services.extra_checks import camera_blocked
        ok = {"mean": 120, "std": 40, "sharp": 300}
        dark = {"mean": 8, "std": 3, "sharp": 1}
        stats = [(i * 0.5, dark if 10 <= i <= 16 else ok) for i in range(30)]
        assert [e["type"] for e in camera_blocked(stats)] == ["camera_blocked"]
        assert camera_blocked([(i * 0.5, ok) for i in range(30)]) == []
        assert camera_blocked([(i * 0.5, dark if i == 5 else ok) for i in range(30)]) == []   # бір кадр — шу

    def test_camera_moved(self):
        from app.services.extra_checks import camera_moved
        rng = np.random.default_rng(0)
        base = rng.uniform(0, 255, (212, 116)).astype(np.float32)
        base = __import__("cv2").GaussianBlur(base, (7, 7), 0)
        moved = np.roll(base, 25, axis=1)
        seq = [(i * 0.5, base if i < 12 else moved) for i in range(30)]
        assert [e["type"] for e in camera_moved(seq)] == ["camera_moved"]
        assert camera_moved([(i * 0.5, base) for i in range(30)]) == []

    def _away_frames(self, visitor_from, visitor_to):
        """Кассир 0–10 с отырады, 10 с тұрып кетеді; қонақ visitor_from–visitor_to аралығында бөлмеде."""
        fr = []
        for i in range(60):
            t = i * 0.5
            ps = []
            if t < 10: ps.append(_person(1, [100, 300, 200, 400]))
            elif t < 12: ps.append(_person(1, [100, 100, 180, 400]))
            if visitor_from <= t <= visitor_to: ps.append(_person(2, [300, 120, 380, 330]))
            fr.append({"t": t, "idx": i, "persons": ps, "objs": [{"cls": "laptop", "conf": 0.9, "box": [120, 330, 180, 380]}]})
        return fr

    def test_left_with_client(self):
        from app.services.extra_checks import left_with_client
        fr = self._away_frames(0, 30)
        assert [e["type"] for e in left_with_client(self._scene(fr, 30.0))] == ["left_with_client"]

    def test_entered_while_away(self):
        from app.services.extra_checks import entered_while_away, left_with_client
        fr = self._away_frames(18, 30)
        sc = self._scene(fr, 30.0)
        assert [e["type"] for e in entered_while_away(sc)] == ["entered_while_away"]
        assert left_with_client(sc) == []

    def test_outsider_near_cashier(self):
        from app.services.extra_checks import outsider_near_cashier
        fr = []
        for i in range(40):
            t = i * 0.5
            ps = [_person(1, [100, 200, 200, 420])]
            ps.append(_person(2, [160, 195, 260, 415]) if 8 <= t <= 14 else _person(2, [330, 200, 430, 420]))
            fr.append({"t": t, "idx": i, "persons": ps, "objs": []})
        assert [e["type"] for e in outsider_near_cashier(self._scene(fr))] == ["outsider_near_cashier"]

    def test_client_across_desk_is_not_outsider(self):
        from app.services.extra_checks import outsider_near_cashier
        fr = [{"t": i * 0.5, "idx": i, "objs": [],
               "persons": [_person(1, [100, 200, 200, 420]), _person(2, [330, 200, 430, 420])]} for i in range(40)]
        assert outsider_near_cashier(self._scene(fr)) == []
