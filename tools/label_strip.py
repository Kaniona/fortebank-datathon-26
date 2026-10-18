"""Кадры вокруг метки: label_strip.py <video> [offsets...] -> review/miss/<video>.jpg"""
import glob, os, sys, cv2, numpy as np
sys.path.insert(0, "tools")
from common import norm
from read_labels import read_labels
L = {n: ts for n, ts, v in read_labels()}
name = norm(sys.argv[1]); offs = [float(x) for x in sys.argv[2:]] or [-4, -2, -1, 0, 1, 2, 4]
p = [q for q in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(q)[:-4]) == name][0]
cap = cv2.VideoCapture(p); fps = cap.get(5); dur = cap.get(7) / fps; tiles = []
for o in offs:
    t = min(max(0, L[name] + o), dur - 0.1)
    cap.set(1, int(t * fps)); ok, fr = cap.read()
    fr = fr[120:640]; fr = cv2.resize(fr, (300, int(fr.shape[0] * 300 / fr.shape[1])))
    cv2.putText(fr, f"{t:.1f}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2); tiles.append(fr)
os.makedirs("review/miss", exist_ok=True); cv2.imwrite(f"review/miss/{name}.jpg", np.hstack(tiles)); print(name, L[name])
