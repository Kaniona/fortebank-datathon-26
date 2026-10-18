"""Прогон живого пайплайна на видео: run_pipeline.py <файл|имя Х-NN> — печатает итог без base64."""
import glob, json, logging, os, sys, time
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
from app.services.analysis_pipeline import AnalysisPipeline, warm_up
arg = sys.argv[1]
path = arg if os.path.exists(arg) else [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == norm(arg)][0]
t = time.time(); warm_up(); print(f"warm-up {time.time()-t:.1f}s")
t = time.time()
res = AnalysisPipeline().analyze(path, progress=lambda p, s: None)
print(f"analyze {time.time()-t:.1f}s verdict={res['verdict']} info={res['video_info']}")
for v in res["violations"]:
    print(" -", v["type"], v["timecode_sec"], v["confidence"], "|", v["description"][:140], "| frames:", [f["timestamp_sec"] for f in v["evidence_frames"]])
    for i, f in enumerate(v["evidence_frames"]):
        import base64
        os.makedirs("review/out", exist_ok=True)
        open(f"review/out/{norm(os.path.basename(path)[:-4])}_{v['type'][:6]}_{i}.jpg", "wb").write(base64.b64decode(f["image_base64"]))
