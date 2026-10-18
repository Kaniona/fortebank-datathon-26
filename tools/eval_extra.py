"""Жаңа тексерістердің кэштегі 42 видеодағы срабатываниесі (live-ке ұқсас: заттар әр 3-кадрда)."""
import glob, json, os, sys
import cv2
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from read_labels import read_labels
from app.services.behavior import analyze_scene
from app.services import extra_checks as X

L = {n: (ts, v) for n, ts, v in read_labels()}
OBJ_EVERY = int(os.environ.get("OBJ_EVERY", "3"))
stats_cache = "cache/framestats.json"
SC = json.load(open(stats_cache)) if os.path.exists(stats_cache) else {}
tot = {}
for n in sorted(L, key=lambda x: int(x.split("-")[1])):
    d = json.load(open(f"cache/{n}.json")); dur = d["n"] / d["fps"]
    frames = [dict(f, objs=(f["objs"] if i % OBJ_EVERY == 0 else [])) for i, f in enumerate(d["frames"])]
    sc = analyze_scene(frames, dur, d["w"], d["h"])
    if n not in SC:
        cap = cv2.VideoCapture([p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]); out = []
        for f in d["frames"]:
            cap.set(1, f["idx"]); ok, fr = cap.read()
            if ok: out.append((f["t"], X.frame_stats(fr)))
        SC[n] = out
    ev = []
    ev += X.phone_in_hand(frames, sc)
    ev += X.camera_blocked([(t, s) for t, s in SC[n]])
    ev += X.left_with_client(sc)
    ev += X.entered_while_away(sc)
    ev += X.outsider_near_cashier(sc)
    gp = f"cache/grays_{n}.npz"
    if not os.path.exists(gp):
        cap = cv2.VideoCapture([p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]); G = []; T = []
        for f in d["frames"]:
            cap.set(1, f["idx"]); ok, fr = cap.read()
            if ok: G.append(X.small_gray(fr)); T.append(f["t"])
        import numpy as np; np.savez_compressed(gp, g=np.stack(G), t=np.array(T))
    import numpy as np; z = np.load(gp)
    ev += X.camera_moved(list(zip(z["t"].tolist(), list(z["g"]))))
    ts, viol = L[n]
    for e in ev:
        k = e["type"]; tot.setdefault(k, {"V": [], "ok": []})["V" if viol else "ok"].append(n)
    if ev: print(f"{n:5} {'V@%d' % ts if viol else 'ok':>6} -> {[(e['type'], round(e['t'])) for e in ev]}")
json.dump(SC, open(stats_cache, "w"))
print("\nҚОРЫТЫНДЫ (V = нарушениелі видеода, ok = таза видеода = ЖАЛҒАН):")
for k, v in tot.items(): print(f"  {k:22} V={len(v['V'])} {v['V']}   ЖАЛҒАН={len(v['ok'])} {v['ok']}")
