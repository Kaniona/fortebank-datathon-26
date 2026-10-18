"""Кропы людей по бокс-трекам: person_crops.py <video> t1 t2 ... -> review/close/pc_<video>.jpg"""
import glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from app.services.tracks import build_tracks, merge_fragments, real_tracks, pick_cashier
name = norm(sys.argv[1]); times = [float(x) for x in sys.argv[2:]]
d = json.load(open(f"cache/{name}.json"))
path = [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == name][0]
cap = cv2.VideoCapture(path); fps = d["fps"]; tiles = []
for t in times:
    fr_rec = min(d["frames"], key=lambda f: abs(f["t"] - t))
    cap.set(1, int(t * fps)); ok, fr = cap.read()
    persons = [p for p in fr_rec["persons"] if p["box"][3] - p["box"][1] > 90]
    if not ok or not persons: tiles.append(np.zeros((420, 300, 3), np.uint8)); continue
    p = max(persons, key=lambda q: q["box"][3] - q["box"][1])
    x1, y1, x2, y2 = p["box"]; w = x2 - x1; h = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(w, h) * 0.75
    crop = fr[int(max(0, cy - s)):int(min(fr.shape[0], cy + s)), int(max(0, cx - s)):int(min(fr.shape[1], cx + s))]
    crop = cv2.resize(crop, (300, 300)); cv2.putText(crop, f"{t:.1f}s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    tiles.append(crop)
os.makedirs("review/close", exist_ok=True); out = f"review/close/pc_{name}.jpg"; cv2.imwrite(out, np.hstack(tiles)); print(out)
