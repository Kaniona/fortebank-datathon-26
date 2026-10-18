"""Правило «ПК не заблокирован при уходе кассира»: проверка по всем кэшированным видео."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from read_labels import read_labels
from laptop_screen import screen_series
from app.services.tracks import build_tracks, merge_fragments, real_tracks, pick_cashier

labels = {n: (ts, v) for n, ts, v in read_labels()}
names = sorted([n for n in labels if os.path.exists(f"cache/{n}.json")], key=lambda x: int(x.split("-")[1]))
print(f"{'video':6}{'label':>9} {'laptop':>6} {'V0':>4} {'Vmax':>5} {'Vend':>5} {'dim@':>5} {'cash_away':>10}")
for n in names:
    r = screen_series(n)
    d = json.load(open(f"cache/{n}.json")); dur = d["n"] / d["fps"]
    tr = merge_fragments(build_tracks(d["frames"])); real, _ = real_tracks(tr); cash = pick_cashier(real)
    away = f"{cash.t1:.0f}-{dur:.0f}" if cash is not None and cash.t1 < dur - 2.5 else "-"
    lab = f"V@{labels[n][0]:.0f}" if labels[n][1] else "ok"
    if r is None:
        print(f"{n:6}{lab:>9} {'none':>6}"); continue
    med, ser = r; v = np.array([s[2] for s in ser]); t = np.array([s[0] for s in ser])
    runmax = np.maximum.accumulate(v); dim = np.where(v < 0.85 * runmax)[0]
    dimt = f"{t[dim[0]]:.0f}" if len(dim) else "-"
    print(f"{n:6}{lab:>9} {'yes':>6} {v[0]:4d} {v.max():5d} {v[-1]:5d} {dimt:>5} {away:>10}")
