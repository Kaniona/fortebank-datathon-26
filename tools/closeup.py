"""Крупные кадры вокруг метки: closeup.py Х-1 t1 t2 t3 ... -> scratchpad/close/<name>.jpg (кроп на рабочую зону)"""
import sys, glob, os, cv2, numpy as np
sys.path.insert(0, "tools")
from common import norm
name = norm(sys.argv[1]); times = [float(x) for x in sys.argv[2:]]
f = [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p).rsplit(".",1)[0]) == name][0]
cap = cv2.VideoCapture(f); fps = cap.get(5); tiles = []
for t in times:
    cap.set(1, int(t*fps)); ok, fr = cap.read()
    fr = fr[100:700, :]               # убираем пустой пол
    fr = cv2.resize(fr, None, fx=1.0, fy=1.0)
    cv2.putText(fr, f"{t:.1f}s", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2); tiles.append(fr)
os.makedirs("review/close", exist_ok=True); out = f"review/close/{name}.jpg"
cv2.imwrite(out, np.hstack(tiles)); print(out)
