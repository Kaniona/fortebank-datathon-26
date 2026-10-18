"""Контактный лист: кадры видео в заданных секундах -> одна картинка."""
import sys, cv2, numpy as np

def sheet(path, times, out, w=300):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
        if not ok: f = np.zeros((848, 464, 3), np.uint8)
        h = int(f.shape[0] * w / f.shape[1]); f = cv2.resize(f, (w, h))
        cv2.putText(f, f"{t:.0f}s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(f)
    cv2.imwrite(out, np.hstack(tiles))

if __name__ == "__main__":
    sheet(sys.argv[1], [float(x) for x in sys.argv[3:]], sys.argv[2])
