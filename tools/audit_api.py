"""ТЗ 10-бөлімі бойынша API контрактын тексереді (жюри сияқты: POST -> job_id -> GET -> JSON)."""
import base64, json, sys, time, glob, os, threading, urllib.request, uuid
URL, KEY = sys.argv[1].rstrip("/"), sys.argv[2]
files = sys.argv[3:]

def post(path):
    b = uuid.uuid4().hex; data = open(path, "rb").read()
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\nContent-Type: video/quicktime\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(URL + "/api/v1/analyze", data=body, headers={"X-API-Key": KEY, "Content-Type": f"multipart/form-data; boundary={b}"})
    t = time.time(); r = urllib.request.urlopen(req, timeout=120); return json.load(r), time.time() - t

def get(job):
    req = urllib.request.Request(f"{URL}/api/v1/jobs/{job}", headers={"X-API-Key": KEY})
    return json.load(urllib.request.urlopen(req, timeout=60))

def check(res, name):
    errs = []
    if res.get("status") != "done": return [f"status={res.get('status')}"]
    if res.get("verdict") not in ("violation", "no_violation"): errs.append("verdict")
    if not isinstance(res.get("violations"), list): errs.append("violations not list")
    if (res["verdict"] == "violation") != bool(res["violations"]): errs.append("verdict/violations mismatch")
    for v in res["violations"]:
        for k, t in (("type", str), ("timecode_sec", (int, float)), ("description", str), ("evidence_frames", list)):
            if not isinstance(v.get(k), t): errs.append(f"{k} type")
        if not (1 <= len(v.get("evidence_frames", [])) <= 3): errs.append("frames count")
        for f in v.get("evidence_frames", []):
            if not isinstance(f.get("timestamp_sec"), (int, float)): errs.append("frame ts")
            try:
                raw = base64.b64decode(f["image_base64"], validate=True)
                if raw[:2] != b"\xff\xd8": errs.append("not JPEG")
            except Exception: errs.append("base64")
    return errs

results = {}
def run(path):
    name = os.path.basename(path); t0 = time.time()
    try:
        j, up = post(path); job = j["job_id"]; assert j.get("status") == "processing", j
        while True:
            r = get(job)
            if r["status"] != "processing": break
            if time.time() - t0 > 330: r = {"status": "timeout"}; break
            time.sleep(2)
        results[name] = (r, time.time() - t0, up)
    except Exception as e:
        results[name] = ({"status": f"EXC {e}"}, time.time() - t0, 0)

th = [threading.Thread(target=run, args=(f,)) for f in files]
t_all = time.time()
for t in th: t.start()
for t in th: t.join()
print(f"\nбарлығы {time.time()-t_all:.0f} с, {len(files)} видео бір мезетте")
bad = 0
for n, (r, dt, up) in results.items():
    e = check(r, n); bad += bool(e) or dt > 300
    print(f"  {n:12} {r.get('status'):10} verdict={r.get('verdict')} violations={len(r.get('violations') or [])} жүктеу={up:.1f}с жалпы={dt:.0f}с {'⚠ LIMIT>300' if dt>300 else ''} {'КОНТРАКТ ҚАТЕ: '+str(e) if e else 'контракт OK'}")
print("НӘТИЖЕ:", "БАРЛЫҒЫ СӘЙКЕС" if not bad else f"{bad} мәселе")
