import pickle, sys
import numpy as np
sys.path.insert(0, "tools")
from read_labels import read_labels
from static_change import timeline

def sustained(tl, level=0.02, hold=5.0, frac=0.85, glob_max=0.4):
    tl = [(t, v) for t, v, gf in tl if gf < glob_max]       # кадры с глобальной сменой освещения не используем
    if not tl: return None
    ts = np.array([t for t, _ in tl]); v = np.array([x for _, x in tl])
    for i, t in enumerate(ts):
        if v[i] < level: continue
        if ts[-1] - t < hold * 0.8: break
        w = (ts >= t) & (ts <= t + hold)
        if (v[w] >= level).mean() >= frac: return float(t), round(float(np.median(v[w])), 3)
    return None

if __name__ == "__main__":
    L = {n: (ts, v) for n, ts, v in read_labels()}
    cache = {}
    for n in sorted(L, key=lambda x: int(x.split("-")[1])):
        tl = timeline(n); cache[n] = tl
        r = sustained(tl); lab = f"V@{L[n][0]:.0f}" if L[n][1] else "ok"
        print(f"{n:5}{lab:>7} max={max(v for _, v, g in tl):.3f} sustained={r}", flush=True)
    pickle.dump(cache, open("cache/static_tl.pkl", "wb"))
