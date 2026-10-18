"""Совпадают ли метки нарушений по времени с событиями: вход/выход человека, начало ходьбы кассира."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from read_labels import read_labels
from app.services.tracks import build_tracks, merge_fragments, real_tracks, pick_cashier

def events(name):
    d = json.load(open(f"cache/{name}.json")); dur = d["n"] / d["fps"]
    tr = merge_fragments(build_tracks(d["frames"])); real, _ = real_tracks(tr); cash = pick_cashier(real)
    ev = []
    for t in real:
        if t.t0 > 1.0: ev.append((t.t0, "enter", t is cash))
        if t.t1 < dur - 1.0: ev.append((t.t1, "exit", t is cash))
    if cash is not None and len(cash.ts) > 4:
        c = cash.centers(); h = cash.height; ts = np.array(cash.ts)
        sp = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-3) / h   # высот/сек
        walking = sp > 0.35
        for i in range(1, len(walking)):
            if walking[i] and not walking[i-1]: ev.append((ts[i], "walk_start", True))
            if (not walking[i]) and walking[i-1]: ev.append((ts[i], "walk_stop", True))
    return sorted(ev), dur

if __name__ == "__main__":
    labels = {n: (ts, v) for n, ts, v in read_labels()}
    gaps = []
    for n in sorted(labels, key=lambda x: int(x.split("-")[1])):
        if not os.path.exists(f"cache/{n}.json"): continue
        ev, dur = events(n); ts, v = labels[n]
        s = " ".join(f"{t:.0f}{k[0]}{'C' if c else ''}" for t, k, c in ev)
        if v:
            near = min((abs(t - ts), t, k) for t, k, c in ev) if ev else (99, 0, "-")
            gaps.append(near[0]); print(f"{n:5} V@{ts:4.0f} dur={dur:4.0f} nearest={near[0]:4.1f}s ({near[2]}) | {s}")
        else:
            print(f"{n:5} ok         dur={dur:4.0f} | {s}")
    print("\nрасстояние до ближайшего события, сек:", np.round(sorted(gaps), 1), " медиана:", np.median(gaps))
