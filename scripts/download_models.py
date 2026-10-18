"""
Однократная загрузка весов в models/ (нужен интернет ТОЛЬКО для этого шага; после него сервис работает офлайн).
    python scripts/download_models.py
"""
import shutil
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(exist_ok=True)

YOLO = {
    "yolov8m-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-pose.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt",
}
for name, url in YOLO.items():
    dst = MODELS / name
    if dst.exists():
        print("есть:", dst.name)
        continue
    print("скачиваю", url)
    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:  # nosec B310 — фиксированный https-адрес
        shutil.copyfileobj(r, f)

OWL = MODELS / "owlv2-base"
if (OWL / "model.safetensors").exists():
    print("есть: owlv2-base")
else:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    name = "google/owlv2-base-patch16-ensemble"
    print("скачиваю", name)
    Owlv2Processor.from_pretrained(name).save_pretrained(OWL)            # nosec B615 — разовая загрузка весов
    Owlv2ForObjectDetection.from_pretrained(name).save_pretrained(OWL)   # nosec B615
print("готово:", MODELS)
