
import csv, random
from collections import Counter

INPUT_FILE = "C:/Users/dhruv/Downloads/Skin_images_examples_06192026-20260623T191520Z-3-001/Skin_Images_examples_06192026/ROI61_CellMap.csv"


X_COL, Y_COL, TYPE_COL = "jm", "im", "CellType"

MICRONS_PER_UNIT = 1.0


TYPE_MAP = {
    "Keratinocyte":      "keratinocyte",
    "M2 Macrophage":     "M2_macrophage",
    "Myofibroblast":     "myofibroblast",
    "Fibroblast":        "fibroblast",
    "CD4+ T cell":       "CD4_T_cell",
    "CD8+ T cell":       "CD8_T_cell",
    "PDPN+ Fibroblasts": "PDPN_fibroblast",
    "Neutrophil":        "neutrophil",
    # "undefined" is omitted
}

DOMAIN     = {"x_min": -550, "x_max": 550, "y_min": -550, "y_max": 550}
MAX_CELLS  = None
OUTPUT_CSV = "C:/Users/dhruv/Downloads/PhysiCell/config/cells.csv"


# read the files
xs, ys, types, dropped = [], [], [], 0
with open(INPUT_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for col in (X_COL, Y_COL, TYPE_COL):
        if col not in reader.fieldnames:
            raise SystemExit(f"[error] column '{col}' not found. File has: {reader.fieldnames}")
    for row in reader:
        t = TYPE_MAP.get(row[TYPE_COL])
        if t is None:
            dropped += 1
            continue
        xs.append(float(row[X_COL]) * MICRONS_PER_UNIT)
        ys.append(float(row[Y_COL]) * MICRONS_PER_UNIT)
        types.append(t)

if dropped:
    print(f"[warning] dropped {dropped} cells whose type wasn't in TYPE_MAP")
if not xs:
    raise SystemExit("[error] no cells matched TYPE_MAP; check the labels and column names.")


cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
xs = [x - cx for x in xs]
ys = [y - cy for y in ys]


span = max(max(xs) - min(xs), max(ys) - min(ys))
box  = min(DOMAIN["x_max"] - DOMAIN["x_min"], DOMAIN["y_max"] - DOMAIN["y_min"]) * 0.95
if span > box:
    s = box / span
    xs = [x * s for x in xs]
    ys = [y * s for y in ys]
    print(f"[note] tissue rescaled by {s:.3f} to fit the domain")

rows = list(zip(xs, ys, [0.0] * len(xs), types))


if MAX_CELLS and len(rows) > MAX_CELLS:
    rows = random.Random(0).sample(rows, MAX_CELLS)
    print(f"[note] downsampled to {MAX_CELLS} cells")


with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y", "z", "type"])
    w.writerows(rows)

print(f"Wrote {len(rows)} cells -> {OUTPUT_CSV}")
for t, c in sorted(Counter(r[3] for r in rows).items()):
    print(f"  {t:18s} {c}")