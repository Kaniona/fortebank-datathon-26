"""Читает data/open/open_labels.xlsx (inlineStr) -> список (video, timestamp_sec|None, is_violation)."""
import re
import sys
sys.path.insert(0, "tools")
from common import norm
import zipfile
from pathlib import Path


def read_labels(path="data/open/open_labels.xlsx"):
    z = zipfile.ZipFile(path)
    x = z.read("xl/worksheets/sheet1.xml").decode()
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", x, flags=re.S):
        cells = {}
        for ref, body in re.findall(r'<c r="([A-Z]+)\d+"[^>]*>(.*?)</c>', row, flags=re.S):
            cells[ref] = "".join(re.findall(r"<t[^>]*>(.*?)</t>", body, flags=re.S)).strip()
        rows.append(cells)
    out = []
    for c in rows[1:]:
        name = norm(c.get("A", ""))  # latin X -> cyrillic Х (как в файлах)
        m = re.search(r"\d+", c.get("B", ""))
        ts = float(m.group()) if m else None
        out.append((name, ts, c.get("C", "").startswith("нарушение")))
    return out


if __name__ == "__main__":
    for r in read_labels():
        print(r)
