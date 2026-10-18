"""Оценка app.services.door по всем видео (кадры из видео, люди из кэша)."""
import glob, json, os, sys
import cv2
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from read_labels import read_labels
from app.services.door import prep, door_timeline, door_left_open

def vpath(n): return [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]

L = {n: (ts, v) for n, ts, v in read_labels()}
for n in sorted(L, key=lambda x: int(x.split("-")[1])):
    d = json.load(open(f"cache/{n}.json")); cap = cv2.VideoCapture(vpath(n)); grays = []; frs = []
    for f in d["frames"]:
        cap.set(1, f["idx"]); ok, fr = cap.read()
        if ok: grays.append(prep(fr)); frs.append(f)
    r = door_left_open(door_timeline(grays, frs)); lab = f"V@{L[n][0]:.0f}" if L[n][1] else "ok"
    flag = "" if (r is not None) == L[n][1] or r is None else "  <-- FALSE POSITIVE"
    print(f"{n:5}{lab:>7} -> {r}{flag}", flush=True)
