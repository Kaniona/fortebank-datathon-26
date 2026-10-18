"""Яркость/«белизна» экрана ноутбука по времени + присутствие кассира у стола."""
import glob, json, sys, os
import cv2, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from read_labels import read_labels

def video_path(name):
    return [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == name][0]

def screen_series(name):
    d = json.load(open(f"cache/{name}.json")); cap = cv2.VideoCapture(video_path(name)); fps = d["fps"]
    lap = [o["box"] for f in d["frames"] for o in f["objs"] if o["cls"] == "laptop" and o["conf"] >= .5]
    if not lap: return None
    med = np.median(np.array(lap), axis=0); x1, y1, x2, y2 = [int(v) for v in med]
    out = []
    for f in d["frames"]:
        cap.set(1, f["idx"]); ok, fr = cap.read()
        if not ok: continue
        roi = fr[max(0,y1):y2, max(0,x1):x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = ((hsv[...,2] > 170) & (hsv[...,1] < 60)).mean()
        dark = (hsv[...,2] < 60).mean(); mean_v = hsv[...,2].mean()
        out.append((f["t"], round(float(white),2), round(float(mean_v)), len(f["persons"])))
    return med, out

if __name__ == "__main__":
    labels = {n: (ts, v) for n, ts, v in read_labels()}
    for n in sys.argv[1:]:
        r = screen_series(n)
        if r is None: print(n, "no laptop"); continue
        med, ser = r
        print(f"== {n} label={labels[n]} laptop_box={[int(v) for v in med]}")
        print("  " + " ".join(f"{t:.0f}:{w:.2f}/{v}" for t, w, v, p in ser[::2]))
