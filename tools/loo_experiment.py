"""Эксперимент: есть ли в агрегатных признаках сигнал нарушения? Leave-one-video-out кросс-валидация."""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from read_labels import read_labels
from app.services.tracks import build_tracks, merge_fragments, real_tracks, pick_cashier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def feats(name):
    d = json.load(open(f"cache/{name}.json")); fr = d["frames"]; dur = d["n"] / d["fps"]
    tr = merge_fragments(build_tracks(fr)); real, _ = real_tracks(tr); cash = pick_cashier(real)
    F = {"dur": dur, "n_real": len(real), "entries": sum(t.t0 > 1.5 for t in real), "exits": sum(t.t1 < dur - 2 for t in real)}
    ts = np.array([f["t"] for f in fr])
    cnt = np.array([sum(1 for t in real if t.t0 - .8 <= x <= t.t1 + .8) for x in ts])
    F["max_simul"] = cnt.max(); F["frac_ge2"] = (cnt >= 2).mean(); F["frac_ge3"] = (cnt >= 3).mean()
    F["others_sec"] = sum(t.duration for t in real if t is not cash) / dur
    if cash is not None and len(cash.ts) > 4:
        c = cash.centers(); h = cash.height; t_ = np.array(cash.ts)
        sp = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(t_), 1e-3) / h
        anchor = np.median(c, axis=0); away = np.linalg.norm(c - anchor, axis=1) / h
        F["walk_frac"] = (sp > .35).mean(); F["path_len"] = np.nansum(np.minimum(sp, 3) * np.diff(t_)) / dur
        F["away_frac"] = (away > .8).mean(); F["away_max"] = away.max()
        ank = np.array([np.mean([p["kpts"][15][2], p["kpts"][16][2]]) for p in cash.persons])
        F["stand_frac"] = (ank > .4).mean()
        wr = np.array([[p["kpts"][9][:2], p["kpts"][10][:2]] for p in cash.persons], float)
        F["wrist_motion"] = float(np.mean(np.linalg.norm(np.diff(wr, axis=0), axis=2) / h)) if len(wr) > 1 else 0
        F["cash_cover"] = cash.duration / dur
    else:
        for k in ("walk_frac", "path_len", "away_frac", "away_max", "stand_frac", "wrist_motion", "cash_cover"): F[k] = 0
    for k, cl in (("phone", ("cell phone",)), ("cup", ("cup", "bottle", "wine glass", "bowl", "banana", "apple", "orange", "donut", "pizza")), ("bag", ("handbag", "backpack", "suitcase"))):
        F[k] = np.mean([any(o["cls"] in cl and o["conf"] >= .3 for o in f["objs"]) for f in fr])
    return F

if __name__ == "__main__":
    labels = {n: v for n, ts, v in read_labels()}
    names = [n for n in labels if os.path.exists(f"cache/{n}.json")]
    X = []; y = np.array([labels[n] for n in names], int)
    for n in names: X.append(feats(n))
    keys = list(X[0]); Xm = np.array([[f[k] for k in keys] for f in X], float)
    print(f"видео: {len(names)}  нарушений: {y.sum()}  базовая точность (мажор. класс): {max(y.mean(), 1-y.mean()):.2f}")
    for nm, mk in (("LogReg", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=500))),
                   ("RF d3", lambda: RandomForestClassifier(200, max_depth=3, random_state=0, min_samples_leaf=2))):
        pred = np.zeros(len(y))
        for i in range(len(y)):
            tr = np.arange(len(y)) != i; m = mk().fit(Xm[tr], y[tr]); pred[i] = m.predict_proba(Xm[i:i+1])[0, 1]
        acc = ((pred > .5) == y).mean(); 
        from sklearn.metrics import roc_auc_score
        print(f"{nm}: LOO accuracy={acc:.2f}  AUC={roc_auc_score(y, pred):.2f}")
