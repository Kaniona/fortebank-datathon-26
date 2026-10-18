"""Кэш OWLv2 по расписанию app.services.probes -> cache/owl_<name>.json (возобновляемо)."""
import glob, json, os, sys, time
import cv2
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from common import norm
from app.services.behavior import analyze_scene
from app.services.probes import plan_probes
from app.services.owl import OwlScorer

def main(names, threads):
    owl = OwlScorer(threads=threads)
    for n in names:
        src = f"cache/{n}.json"; out = f"cache/owl_{n}.json"
        if not os.path.exists(src): continue
        d = json.load(open(src)); dur = d["n"] / d["fps"]
        sc = analyze_scene(d["frames"], dur, d["w"], d["h"]); plan = plan_probes(sc)
        path = [p for p in glob.glob("data/open/videos/*.mov") if norm(os.path.basename(p)[:-4]) == n][0]
        res = json.load(open(out)) if os.path.exists(out) else {"probes": []}
        done = {(p["t"], p["kind"]) for p in res["probes"]}
        cap = cv2.VideoCapture(path); t0 = time.time()
        for pr in plan:
            if (pr.t, pr.kind) in done: continue
            cap.set(1, int(pr.t * d["fps"])); ok, fr = cap.read()
            if not ok: continue
            x1, y1, x2, y2 = pr.roi; crop = fr[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0: continue
            r = owl.score(crop, pr.queries)
            res["probes"].append({"t": pr.t, "kind": pr.kind, "roi": pr.roi, "scores": {q: [round(s, 3), b] for q, (s, b) in r.items()}})
        tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
        print(f"{n}: {len(plan)} probes in {time.time()-t0:.0f}s away={[(round(a),round(b)) for a,b in sc.away]}", flush=True)

if __name__ == "__main__":
    names = sys.argv[2:] or sorted(os.path.basename(f)[:-5] for f in glob.glob("cache/*.json") if "/owl_" not in f)
    main(names, int(sys.argv[1]))
