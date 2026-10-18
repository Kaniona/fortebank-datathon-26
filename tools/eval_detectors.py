"""Оценка детекторов на кэше: по каждому видео — что сработало и совпадает ли с меткой."""
import glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from read_labels import read_labels
from app.services.behavior import analyze_scene
from app.services.state_checks import laptop_brightness, pc_left_unlocked

# --- пороги OWL (подбираются по датасету) ---
OWL_TH = {"a bundle of banknotes": 0.20, "a tablet computer": 0.30, "a handbag": 0.30, "a green bag": 0.25}

def vpath(n): return [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]

def laptop_series(n, d, sc):
    if sc.laptop_box is None: return []
    cap = cv2.VideoCapture(vpath(n)); out = []
    for f in d["frames"]:
        cap.set(1, f["idx"]); ok, fr = cap.read()
        if ok:
            v = laptop_brightness(fr, sc.laptop_box)
            if v is not None: out.append((f["t"], v))
    return out

def owl_items(n, sc):
    p = f"cache/owl_{n}.json"
    if not os.path.exists(p): return None
    pr = json.load(open(p))["probes"]; hits = {}
    for t_leave, t_end in sc.away:
        for ts, q in pr.items():
            t = float(ts)
            if t_leave + 1.0 <= t <= t_end:
                for name, th in OWL_TH.items():
                    if q[name][0] >= th: hits[name] = max(hits.get(name, 0), q[name][0])
    return hits

if __name__ == "__main__":
    labels = {n: (ts, v) for n, ts, v in read_labels()}
    names = sorted([n for n in labels if os.path.exists(f"cache/{n}.json")], key=lambda x: int(x.split("-")[1]))
    for n in names:
        d = json.load(open(f"cache/{n}.json")); dur = d["n"] / d["fps"]
        sc = analyze_scene(d["frames"], dur, d["w"], d["h"])
        lab = f"V@{labels[n][0]:.0f}" if labels[n][1] else "ok"
        reasons = []
        for t0, t1 in sc.away:
            r = pc_left_unlocked(laptop_series(n, d, sc), t0)
            if r: reasons.append(f"PC_UNLOCKED@{t0:.0f}({r['brightness']})")
        it = owl_items(n, sc)
        if it: reasons += [f"{k.split()[-1]}={v:.2f}" for k, v in it.items()]
        print(f"{n:5}{lab:>7} away={[(round(a), round(b)) for a, b in sc.away]} -> {reasons or '-'}")
