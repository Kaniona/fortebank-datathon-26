"""Для каждого видео: положение остальных людей относительно кассира (в высотах кассира)."""
import json, os, sys
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from read_labels import read_labels
from app.services.tracks import build_tracks, merge_fragments, real_tracks, pick_cashier

labels = {n: (ts, v) for n, ts, v in read_labels()}
names = sorted([n for n in labels if os.path.exists(f"cache/{n}.json")], key=lambda x: int(x.split("-")[1]))
for n in names:
    d = json.load(open(f"cache/{n}.json")); dur = d["n"] / d["fps"]
    real, _ = real_tracks(merge_fragments(build_tracks(d["frames"]))); cash = pick_cashier(real)
    lab = f"V@{labels[n][0]:.0f}" if labels[n][1] else "ok"
    print(f"== {n} {lab} dur={dur:.0f}")
    if cash is None: continue
    H = cash.height
    for t in real:
        if t is cash: continue
        offs = []
        for ts, p in zip(t.ts, t.persons):
            c = cash.at(ts, 1.0)
            if c is None: continue
            cx = (c["box"][0] + c["box"][2]) / 2; cy = (c["box"][1] + c["box"][3]) / 2
            px = (p["box"][0] + p["box"][2]) / 2; py = (p["box"][1] + p["box"][3]) / 2
            offs.append(((px - cx) / H, (py - cy) / H))
        if not offs: continue
        o = np.array(offs); m = np.median(o, axis=0); dist = np.linalg.norm(o, axis=1)
        print(f"   other id{t.id:<3} t={t.t0:5.1f}-{t.t1:5.1f} dur={t.duration:5.1f}s med_off=({m[0]:+.2f},{m[1]:+.2f}) min_dist={dist.min():.2f} h_ratio={t.height/H:.2f}")
