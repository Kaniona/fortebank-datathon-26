"""
Планирование и выполнение точечных проверок OWLv2.

Полный перебор кадров OWLv2 на CPU слишком дорог, поэтому проверяются только значимые моменты:
  - рабочий стол, пока кассира нет (ценности без присмотра);
  - рабочий стол и сам кассир, пока он на месте (личные вещи, еда/напитки);
  - кассир в момент входа в помещение (сумка/телефон в руках).
"""
from dataclasses import dataclass

DESK_QUERIES = ["a bundle of banknotes", "a tablet computer", "a handbag", "a jacket or coat", "a cup",
                "food or a snack", "a bottle", "a sheet of paper", "a green bag"]
PERSON_QUERIES = ["a handbag", "a mobile phone", "a cup", "a bag"]


@dataclass
class Probe:
    t: float
    kind: str          # 'desk_away' | 'desk_seated' | 'person'
    roi: list          # [x1, y1, x2, y2] в координатах кадра
    queries: list


def plan_probes(sc, max_probes: int = 9) -> list[Probe]:
    """Строит расписание проб по результатам анализа сцены (analyze_scene)."""
    if sc.cashier is None or sc.desk_roi is None:
        return []
    probes: list[Probe] = []
    # 1. Пока кассира нет у стола
    for t0, t1 in sc.away:
        tv = sc.vacant_from.get((t0, t1), t0)
        # Пробы — когда стол уже пуст: сразу после ухода и в конце интервала
        for t in sorted({round(min(t1, tv + 1.0), 1), round(max(tv, t1 - 0.5), 1)}):
            probes.append(Probe(t, "desk_away", sc.desk_roi, DESK_QUERIES))
    # 2. Пока кассир на месте — три момента
    ts = [t for t in sc.cashier.ts]
    away_set = lambda t: any(a <= t <= b for a, b in sc.away)
    seated_ts = [t for t in ts if not away_set(t)]
    if seated_ts:
        for frac in (0.25, 0.55, 0.85):
            t = min(seated_ts, key=lambda x: abs(x - (seated_ts[0] + frac * (seated_ts[-1] - seated_ts[0]))))
            probes.append(Probe(round(t, 1), "desk_seated", sc.desk_roi, DESK_QUERIES))
    # 3. Вход кассира в помещение
    if sc.cashier.t0 > 1.0:
        t = sc.cashier.t0 + 1.0
        p = sc.cashier.at(t, 1.0)
        if p is not None:
            x1, y1, x2, y2 = p["box"]; w, h = x2 - x1, y2 - y1
            probes.append(Probe(round(t, 1), "person", [int(x1 - 0.2 * w), int(y1 - 0.1 * h), int(x2 + 0.2 * w), int(y2 + 0.1 * h)], PERSON_QUERIES))
    # Лимит: приоритет у проб «стол без кассира»
    order = {"desk_away": 0, "person": 1, "desk_seated": 2}
    probes.sort(key=lambda p: (order[p.kind], p.t))
    return sorted(probes[:max_probes], key=lambda p: p.t)
