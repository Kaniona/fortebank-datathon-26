"""Прогон ЖИВОГО пайплайна по всем видео обучающей выборки (без кэша) и сравнение с разметкой."""
import glob, json, logging, os, sys, time
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from read_labels import read_labels
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from app.services.analysis_pipeline import AnalysisPipeline, warm_up

OUT = "review/batch_results.json"
labels = {n: (ts, v) for n, ts, v in read_labels()}
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
warm_up(); pipe = AnalysisPipeline()
for n in sorted(labels, key=lambda x: int(x.split("-")[1])):
    if n in res: continue
    path = [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]
    t = time.time(); r = pipe.analyze(path)
    res[n] = {"verdict": r["verdict"], "sec": round(time.time() - t, 1), "dur": r["video_info"]["duration_sec"],
              "violations": [{"type": v["type"], "t": v["timecode_sec"], "conf": v["confidence"]} for v in r["violations"]]}
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    ts, viol = labels[n]
    print(f"{'✔' if (r['verdict']=='violation')==viol else '✘'} {n:5} {'V@%d' % ts if viol else 'ok':>6} -> {r['verdict']:12} {res[n]['violations']} {res[n]['sec']}s", flush=True)
tp = sum(1 for n, v in res.items() if v["verdict"] == "violation" and labels[n][1]); fp = sum(1 for n, v in res.items() if v["verdict"] == "violation" and not labels[n][1])
fn = sum(1 for n, v in res.items() if v["verdict"] != "violation" and labels[n][1]); tn = len(res) - tp - fp - fn
print(f"\nвидео={len(res)} accuracy={(tp+tn)/len(res):.2f} TP={tp} FP={fp} TN={tn} FN={fn} max_time={max(v['sec'] for v in res.values())}s")
