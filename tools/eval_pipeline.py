"""
Оффлайн-оценка пайплайна по кэшу (YOLO + OWL): вердикт видео, ошибка таймкода, список сработавших типов.
Использует те же функции, что и живой сервис (behavior / state_checks / build_item_events).
"""
import glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from read_labels import read_labels
from app.services.behavior import analyze_scene
from app.services.state_checks import laptop_brightness, pc_left_unlocked
from app.services.analysis_pipeline import build_item_events

def vpath(n): return [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]

def lap_series(n, d):
    cap = cv2.VideoCapture(vpath(n)); out = []; box = None
    for f in d["frames"]:
        for o in f["objs"]:
            if o["cls"] == "laptop" and o["conf"] >= .5: box = o["box"]
        if box is None: continue
        cap.set(1, f["idx"]); ok, fr = cap.read()
        if ok:
            v = laptop_brightness(fr, box)
            if v is not None: out.append((f["t"], v))
    return out

def run(n):
    d = json.load(open(f"cache/{n}.json")); dur = d["n"] / d["fps"]
    sc = analyze_scene(d["frames"], dur, d["w"], d["h"]); ev = []
    ls = lap_series(n, d)
    for t0, _ in sc.away:
        r = pc_left_unlocked(ls, t0)
        if r: ev.append({"type": "pc_unlocked", "t": t0}); break
    p = f"cache/owl_{n}.json"
    if os.path.exists(p): ev += build_item_events(sc, json.load(open(p))["probes"], d["frames"])
    return sc, ev

if __name__ == "__main__":
    labels = {n: (ts, v) for n, ts, v in read_labels()}
    names = sorted([n for n in labels if os.path.exists(f"cache/{n}.json") and os.path.exists(f"cache/owl_{n}.json")],
                   key=lambda x: int(x.split("-")[1]))
    tp = fp = tn = fn = 0; errs = []
    for n in names:
        sc, ev = run(n); ts, viol = labels[n]; pred = bool(ev)
        tp += pred and viol; fp += pred and not viol; tn += (not pred) and (not viol); fn += (not pred) and viol
        te = ""
        if pred and viol: e = min(abs(x["t"] - ts) for x in ev); errs.append(e); te = f"Δt={e:.0f}с"
        mark = "✔" if pred == viol else "✘"
        print(f"{mark} {n:5} {'V@%d' % ts if viol else 'ok':>6}  -> {[(e['type'], round(e['t'])) for e in ev] or '-'} {te}")
    n_ = tp + fp + tn + fn
    print(f"\nвидео={n_}  точность={(tp+tn)/max(n_,1):.2f}  TP={tp} FP={fp} TN={tn} FN={fn}  "
          f"precision={tp/max(tp+fp,1):.2f} recall={tp/max(tp+fn,1):.2f}  медиана|Δt|={np.median(errs) if errs else float('nan'):.1f}с")
