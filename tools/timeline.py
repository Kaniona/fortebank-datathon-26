"""Печатает по видео: треки людей (после фильтра «людей за стеклом») и метку из разметки."""
import json, sys, glob, os
sys.path.insert(0, "tools")
from read_labels import read_labels

def tracks(d, min_len=3):
    T = {}
    for fr in d["frames"]:
        for p in fr["persons"]:
            T.setdefault(p["id"], []).append((fr["t"], p))
    return {k: v for k, v in T.items() if len(v) >= min_len}

def summarize(name):
    d = json.load(open(f"cache/{name}.json")); T = tracks(d)
    hmax = max(max(p["box"][3] - p["box"][1] for _, p in v) for v in T.values())
    out = []
    for k, v in sorted(T.items(), key=lambda kv: kv[1][0][0]):
        hs = [p["box"][3] - p["box"][1] for _, p in v]; hm = sorted(hs)[len(hs)//2]
        cxs = [(p["box"][0] + p["box"][2]) / 2 for _, p in v]; cys = [(p["box"][1] + p["box"][3]) / 2 for _, p in v]
        move = (max(cxs) - min(cxs)) + (max(cys) - min(cys))
        ghost = hm < 0.45 * hmax
        out.append(f"  id{k:<3} t={v[0][0]:5.1f}-{v[-1][0]:5.1f} n={len(v):3d} h={hm:5.0f} ({hm/hmax:.2f}) move={move:5.0f} {'GHOST' if ghost else ''}")
    return out

if __name__ == "__main__":
    labels = {n: (ts, v) for n, ts, v in read_labels()}
    names = sys.argv[1:] or sorted(os.path.basename(f)[:-5] for f in glob.glob("cache/*.json"))
    for n in names:
        print(f"== {n} label={labels.get(n)}"); print("\n".join(summarize(n)))
