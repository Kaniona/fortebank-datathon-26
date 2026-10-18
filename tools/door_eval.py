"""Есік ашық қалды ма: адам есіктен өткеннен кейін есік аймағы бұрынғы (жабық) күйіне оралды ма."""
import glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from read_labels import read_labels
from app.services.tracks import build_tracks, merge_fragments, real_tracks

def vpath(n): return [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]

def load_gray(n, d):
    cap = cv2.VideoCapture(vpath(n)); out = {}
    for f in d["frames"]:
        cap.set(1, f["idx"]); ok, fr = cap.read()
        if ok: out[f["t"]] = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
    return out

def person_mask(f, shape, pad=0.15):
    m = np.zeros(shape, bool)
    for p in f["persons"]:
        x1, y1, x2, y2 = p["box"]; w, h = x2 - x1, y2 - y1
        m[max(0, int(y1 - pad * h)):int(y2 + pad * h), max(0, int(x1 - pad * w)):int(x2 + pad * w)] = True
    return m

def passages(d):
    dur = d["n"] / d["fps"]
    real, _ = real_tracks(merge_fragments(build_tracks(d["frames"])))
    ev = []
    for t in real:
        if t.t0 > 1.0: ev.append(("in", t.t0, t.persons[0]["box"]))
        if t.t1 < dur - 1.0: ev.append(("out", t.t1, t.persons[-1]["box"]))
    return sorted(ev, key=lambda e: e[1]), dur

def door_scores(n):
    d = json.load(open(f"cache/{n}.json")); ev, dur = passages(d)
    if not ev: return []
    g = load_gray(n, d); fr_by_t = {f["t"]: f for f in d["frames"]}; ts = sorted(g)
    H, W = next(iter(g.values())).shape
    res = []
    for kind, te, box in ev:
        x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
        roi = (int(max(0, x1 - 0.5 * w)), int(max(0, y1 - 0.2 * h)), int(min(W, x2 + 0.5 * w)), int(min(H, y2 + 0.1 * h)))
        rx1, ry1, rx2, ry2 = roi
        def reg(t): return g[t][ry1:ry2, rx1:rx2]
        def free(t): return ~person_mask(fr_by_t[t], (H, W))[ry1:ry2, rx1:rx2]
        pre = [t for t in ts if te - 5 <= t <= te - 1.0 and free(t).mean() > 0.6]
        if not pre: pre = [t for t in ts if t <= max(ts[0] + 1, te - 1.0) and free(t).mean() > 0.6][:4]
        post = [t for t in ts if te + 2.5 <= t <= te + 9 and free(t).mean() > 0.5]
        if len(pre) < 1 or len(post) < 2: res.append((kind, te, None, roi)); continue
        ref = np.median(np.stack([reg(t) for t in pre]), axis=0)
        diffs = []
        for t in post:
            m = free(t); dd = np.abs(reg(t) - ref)[m]
            diffs.append(float((dd > 25).mean()))
        res.append((kind, te, float(np.median(diffs)), roi))
    return res

if __name__ == "__main__":
    L = {n: (ts, v) for n, ts, v in read_labels()}
    for n in sorted(L, key=lambda x: int(x.split("-")[1])):
        r = door_scores(n)
        lab = f"V@{L[n][0]:.0f}" if L[n][1] else "ok"
        s = " ".join(f"{k}@{t:.0f}:{'-' if sc is None else f'{sc:.2f}'}" for k, t, sc, _ in r)
        print(f"{n:5}{lab:>7} | {s}")
