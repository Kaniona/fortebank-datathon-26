"""Считает по каждому видео числовые признаки и печатает таблицу + AUC признака относительно метки."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from read_labels import read_labels
from app.services.tracks import build_tracks, merge_fragments, real_tracks, pick_cashier

def count_at(tracks, t, tol=0.8):
    return sum(1 for tr in tracks if tr.t0 - tol <= t <= tr.t1 + tol)

def features(name):
    d = json.load(open(f"cache/{name}.json")); fr = d["frames"]; dur = d["n"] / d["fps"]
    tr = merge_fragments(build_tracks(fr)); real, ghost = real_tracks(tr)
    ts = [f["t"] for f in fr]
    cnt = np.array([count_at(real, t) for t in ts])
    cash = pick_cashier(real)
    F = {"dur": dur, "n_real": len(real), "max_simul": int(cnt.max()) if len(cnt) else 0,
         "sec_ge3": float((cnt >= 3).sum() / 2), "sec_ge2": float((cnt >= 2).sum() / 2), "sec_eq0": float((cnt == 0).sum() / 2)}
    F["entries"] = sum(1 for t in real if t.t0 > 1.5); F["exits"] = sum(1 for t in real if t.t1 < dur - 2.0)
    if cash:
        F["cash_dur_frac"] = cash.duration / dur
        F["cash_end_gap"] = max(0.0, dur - cash.t1); F["cash_start_gap"] = cash.t0
        gaps = np.diff(cash.ts); F["cash_max_gap"] = float(gaps.max()) if len(gaps) else 0.0
    ph = [sum(1 for o in f["objs"] if o["cls"] == "cell phone" and o["conf"] >= 0.3) for f in fr]
    F["phone_fr"] = float(np.sum(np.array(ph) > 0)) / 2
    for k, cl in (("cup_fr", ("cup", "bottle", "wine glass", "bowl", "banana", "apple", "orange", "donut", "pizza")), ("bag_fr", ("handbag", "backpack", "suitcase"))):
        F[k] = float(sum(1 for f in fr if any(o["cls"] in cl and o["conf"] >= 0.3 for o in f["objs"]))) / 2
    return F

def auc(x, y):
    x = np.array(x, float); y = np.array(y, bool)
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))

if __name__ == "__main__":
    labels = {n: (ts, v) for n, ts, v in read_labels()}
    names = [n for n in labels if os.path.exists(f"cache/{n}.json")]
    rows = {n: features(n) for n in names}
    keys = sorted({k for r in rows.values() for k in r})
    print("video label  " + " ".join(f"{k[:9]:>9}" for k in keys))
    for n in names:
        print(f"{n:6} {'V' if labels[n][1] else '-'} @{labels[n][0] or 0:4.0f} " + " ".join(f"{rows[n].get(k, float('nan')):9.2f}" for k in keys))
    if len(names) >= 10:
        y = [labels[n][1] for n in names]
        print("\nAUC (violation vs ok):")
        for k in keys: print(f"  {k:14} {auc([rows[n].get(k, 0) for n in names], y):.2f}")
