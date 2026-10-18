"""Кэш признаков: YOLOv8-pose (+track) и YOLOv8 объекты на 2 fps для каждого видео -> cache/<name>.json"""
import glob, json, os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, "tools")
from common import norm
from ultralytics import YOLO

SAMPLE_FPS = 2.0
OBJ_CLASSES = {67: "cell phone", 73: "book", 41: "cup", 39: "bottle", 24: "backpack", 26: "handbag",
               28: "suitcase", 63: "laptop", 62: "tv", 65: "remote", 46: "banana", 47: "apple", 49: "orange", 54: "donut", 53: "pizza", 45: "bowl", 40: "wine glass", 75: "vase", 76: "scissors", 64: "mouse"}

def main(files, threads):
    torch.set_num_threads(threads)
    pose = YOLO("yolov8m-pose.pt"); det = YOLO("yolov8m.pt")
    os.makedirs("cache", exist_ok=True)
    for f in files:
        name = norm(os.path.basename(f).rsplit(".", 1)[0])
        out = f"cache/{name}.json"
        if os.path.exists(out): continue
        t0 = time.time(); cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = max(1, round(fps / SAMPLE_FPS))
        frames = []; i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % step == 0:
                h, w = fr.shape[:2]
                rp = pose.track(fr, imgsz=960, conf=0.25, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
                persons = []
                if rp.boxes is not None and len(rp.boxes):
                    ids = rp.boxes.id.int().tolist() if rp.boxes.id is not None else [-1] * len(rp.boxes)
                    for k, b in enumerate(rp.boxes):
                        persons.append({"id": ids[k], "box": [round(v, 1) for v in b.xyxy[0].tolist()],
                                        "conf": round(float(b.conf), 3),
                                        "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in rp.keypoints.data[k].tolist()]})
                rd = det(fr, imgsz=960, conf=0.2, classes=list(OBJ_CLASSES), verbose=False)[0]
                objs = [{"cls": OBJ_CLASSES[int(b.cls)], "box": [round(v, 1) for v in b.xyxy[0].tolist()], "conf": round(float(b.conf), 3)} for b in rd.boxes]
                frames.append({"t": round(i / fps, 2), "idx": i, "persons": persons, "objs": objs})
            i += 1
        tmp = out + ".tmp"
        json.dump({"name": name, "fps": fps, "w": w, "h": h, "n": n, "frames": frames}, open(tmp, "w"))
        os.replace(tmp, out)  # атомарная запись: обрыв процесса не оставит битый файл
        print(f"{name}: {len(frames)} frames in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    files = sorted(glob.glob("data/open/videos/*.mov"))
    part, parts, threads = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    main(files[part::parts], threads)
