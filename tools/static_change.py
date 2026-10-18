"""Тұрақты өзгеріс: адамдар (±3 с) жасырылған кадрдың басқы кадрдан айырмасы, жоғарғы бөлік."""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from read_labels import read_labels
from door_eval import load_gray, person_mask

def timeline(n, top=0.55, thr=25, win=3.0, pad=0.3):
    d = json.load(open(f"cache/{n}.json")); g = load_gray(n, d); frames = d["frames"]
    ts = sorted(g); H, W = g[ts[0]].shape; y2 = int(H * top)
    pm = {f["t"]: person_mask(f, (H, W), pad) for f in frames}
    first = ts[:4]
    stack = np.stack([g[t] for t in first]); masks = np.stack([pm[t] for t in first])
    ref = np.ma.median(np.ma.array(stack, mask=masks), axis=0).filled(0); known = ~masks.all(0)
    out = []
    for t in ts:
        m = known.copy()
        for f in frames:
            if abs(f["t"] - t) <= win: m &= ~pm[f["t"]]
        m[y2:] = False
        diff = np.abs(g[t] - ref) > thr
        glob_frac = float((np.abs(g[t] - ref) > thr)[known].mean())
        bm = ((diff & m).astype(np.uint8)) * 255
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(bm)
        best = 0.0
        for k in range(1, nlab):
            x, y, w, h, a = stats[k]
            if h >= 0.30 * H and w >= 0.10 * W:          # дверное полотно: высокое и не нитевидное
                best = max(best, a / (H * W))
        out.append((t, best, glob_frac))
    return out

if __name__ == "__main__":
    L = {n: (ts, v) for n, ts, v in read_labels()}
    for n in sys.argv[1:]:
        tl = timeline(n); lab = f"V@{L[n][0]:.0f}" if L[n][1] else "ok"
        print(f"{n:5}{lab:>7} " + " ".join(f"{t:.0f}:{v:.2f}/{gf:.2f}" for t, v, gf in tl[::2]))
