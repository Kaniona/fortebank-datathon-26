"""Басқа ракурсқа төзімділік: видеоны өзгертіп (аудару, бұру, ұлғайту), нәтиже түпнұсқамен сәйкес пе."""
import glob, json, os, subprocess, sys, time
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from app.services.analysis_pipeline import AnalysisPipeline, warm_up

orig = json.load(open("review/batch_results.json"))
NAMES = sys.argv[1].split(",")
T = {  # ракурс өзгерістері
    "flip": "hflip",
    "rot8": "rotate=8*PI/180:ow=iw:oh=ih:fillcolor=black,scale=iw:ih",
    "zoomL": "crop=iw*0.78:ih*0.78:0:ih*0.05,scale=464:848",
    "zoomR": "crop=iw*0.78:ih*0.78:iw*0.22:ih*0.05,scale=464:848",
}
os.makedirs("review/audit", exist_ok=True)
warm_up(); pipe = AnalysisPipeline(); out = {}
for n in NAMES:
    src = [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == norm(n)][0]
    for tn, vf in T.items():
        dst = f"review/audit/{norm(n)}_{tn}.mov"
        if not os.path.exists(dst):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
        t = time.time(); r = pipe.analyze(dst)
        types = [v["type"][:22] for v in r["violations"]]
        same = r["verdict"] == orig[norm(n)]["verdict"]
        print(f"{'=' if same else '≠'} {norm(n):5} {tn:6} orig={orig[norm(n)]['verdict'][:4]} now={r['verdict'][:4]} {types} {time.time()-t:.0f}s", flush=True)
