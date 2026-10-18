"""Для каждого видео делает контактный лист вокруг метки (или равномерно, если без нарушения)."""
import cv2, glob, os, sys
sys.path.insert(0, "tools")
from common import norm
sys.path.insert(0, "tools")
from contact_sheet import sheet
from read_labels import read_labels

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
files = {norm(os.path.basename(f).rsplit(".",1)[0]): f for f in glob.glob("data/open/videos/*.mov")}
for name, ts, viol in read_labels():
    f = files[name]; cap = cv2.VideoCapture(f)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    if viol:
        times = [max(0, ts + d) for d in (-6, -3, -1.5, 0, 1.5, 3, 6)]
    else:
        times = [dur * k / 8 for k in range(1, 8)]
    times = [min(t, dur - 0.1) for t in times]
    sheet(f, times, f"{OUT}/{name}.jpg", w=250)
    print(name, f"dur={dur:.1f}", "VIOL@%s" % ts if viol else "ok")
