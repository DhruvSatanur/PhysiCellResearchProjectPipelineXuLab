import numpy as np, csv, json, sys, argparse
import xml.etree.ElementTree as ET
from collections import Counter

DEFAULT_CONFIG = "cells_config.json"

PALETTE = ["#d1495b","#00798c","#edae49","#66a182","#8b5fbf","#e07a5f",
           "#3d5a80","#81b29a","#5e548e","#9c6644","#e9c46a","#b5838d"]

# PhysiCell default cell radius for volume 2494 um^3. Used when a region
# specifies "packing" instead of an absolute "spacing".
CELL_RADIUS = 8.412710547954228


def load_config(path):
    try:
        with open(path) as f: return json.load(f)
    except FileNotFoundError:
        sys.exit(f"[error] config file not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[error] '{path}' is not valid JSON: {e}\n"
                 "        common causes: trailing comma, missing comma, or unquoted text.")


_color_cache = {}
def get_color(t):
    if t in CELL_TYPES: return CELL_TYPES[t]
    if t not in _color_cache:
        _color_cache[t] = PALETTE[len(_color_cache) % len(PALETTE)]
    return _color_cache[t]


def expand(comp, _seen=None):

    _seen = _seen or set(); out = {}
    for key, frac in comp.items():
        if frac <= 0: continue
        if key in GROUPS:
            if key in _seen: sys.exit(f"[error] circular group reference involving '{key}'")
            sub = expand(GROUPS[key], _seen | {key}); s = sum(sub.values())
            for t, sf in sub.items(): out[t] = out.get(t, 0.0) + frac * sf / s
        else:
            out[key] = out.get(key, 0.0) + frac
    return out


def hex_points(x0, x1, y0, y1, s):
    m = s / 2.0; dy = s * np.sqrt(3) / 2.0; pts = []
    if x1 <= x0 or y1 <= y0: return pts
    for r, y in enumerate(np.arange(y0 + m, y1 - m + 1e-9, dy)):
        off = (s / 2.0) if (r % 2) else 0.0
        for x in np.arange(x0 + m + off, x1 - m + 1e-9, s):
            pts.append((float(x), float(y)))
    return pts


def hex_points_radial(cx, cy, r_in, r_out, s):
    """Hex lattice clipped to a disk (r_in=0) or annulus, centred on (cx, cy).

    Spacing s is centre-to-centre. s = 2*CELL_RADIUS (packing 1.0) reproduces
    PhysiCell Studio's 'hex fill': a disk of R=200 gives 510 cells, matching
    the 511 agents in the Studio run log of Heiland et al. 2024, Fig 9. Note
    PhysiCell's own sample-project snippets use 0.95*2*CELL_RADIUS, which would
    give 565 -- do not assume the sample value if you are matching that paper.
    """
    dy = s * np.sqrt(3) / 2.0; pts = []
    if r_out <= 0 or s <= 0: return pts
    for r, y in enumerate(np.arange(-r_out, r_out + 1e-9, dy)):
        off = (s / 2.0) if (r % 2) else 0.0
        for x in np.arange(-r_out + off, r_out + 1e-9, s):
            d2 = x*x + y*y
            if r_in*r_in <= d2 <= r_out*r_out:
                pts.append((float(cx + x), float(cy + y)))
    return pts


def random_points_radial(cx, cy, r_in, r_out, n, rng):
    """Uniform-by-area random fill of a disk or annulus (Studio's 'random fill').

    The sqrt keeps density uniform in area; sampling r directly would pile
    cells up against the inner edge.
    """
    if n <= 0: return []
    r = np.sqrt(rng.uniform(r_in**2, r_out**2, n))
    t = rng.uniform(0.0, 2.0*np.pi, n)
    return [(float(cx + ri*np.cos(ti)), float(cy + ri*np.sin(ti)))
            for ri, ti in zip(r, t)]


def random_points_rect(x0, x1, y0, y1, n, rng):
    if n <= 0 or x1 <= x0 or y1 <= y0: return []
    return [(float(x), float(y))
            for x, y in zip(rng.uniform(x0, x1, n), rng.uniform(y0, y1, n))]


def assign_types(n, comp, rng):
    if n == 0: return []
    conc = expand(comp); types = list(conc)
    fracs = np.array([conc[t] for t in types], float); fracs /= fracs.sum()
    counts = np.floor(fracs * n).astype(int)
    for i in np.argsort(-(fracs * n - np.floor(fracs * n)))[: n - counts.sum()]: counts[i] += 1
    labels = []
    for i, t in enumerate(types): labels += [t] * counts[i]
    rng.shuffle(labels); return labels


def get_regions():

    if REGIONS: return REGIONS
    regions = []; top = 1.0
    for b in BANDS:
        h = b["height"]
        regions.append({"name": b.get("name", "band"),
                        "x": [0.0, 1.0], "y": [top - h, top],
                        "composition": b["composition"]})
        top -= h
    return regions


def region_abs(reg):

    d = DOMAIN
    fx = reg.get("x", [0.0, 1.0]); fy = reg.get("y", [0.0, 1.0])
    return (d["x_min"] + fx[0]*(d["x_max"]-d["x_min"]), d["x_min"] + fx[1]*(d["x_max"]-d["x_min"]),
            d["y_min"] + fy[0]*(d["y_max"]-d["y_min"]), d["y_min"] + fy[1]*(d["y_max"]-d["y_min"]))


def region_spacing(reg):
    """Region spacing precedence: explicit spacing > packing > global SPACING.

    'packing' is a multiple of the cell diameter, so packing 1.0 means cells
    just touching. Rectangular regions keep the global default so existing
    configs are unaffected.
    """
    if "spacing" in reg: return float(reg["spacing"])
    if "packing" in reg: return float(reg["packing"]) * 2.0 * CELL_RADIUS
    return SPACING


def region_points(reg, rng):
    """Dispatch on shape. Absent 'shape' means rect, so old configs still work.

    Rectangles use FRACTIONAL x/y bounds (0-1 of the domain), as before.
    Radial shapes use ABSOLUTE microns for center/r_inner/r_outer, because a
    fraction of a non-square domain has no single sensible meaning for a radius.
    """
    shape = reg.get("shape", "rect").lower()
    fill  = reg.get("fill", "hex").lower()
    s     = region_spacing(reg)

    if shape in ("disk", "annulus", "circle"):
        cx, cy = reg.get("center", [0.0, 0.0])
        r_in  = float(reg.get("r_inner", 0.0))
        r_out = float(reg["r_outer"])
        if r_in >= r_out:
            sys.exit(f"[error] region '{reg.get('name','?')}': "
                     f"r_inner ({r_in}) must be less than r_outer ({r_out})")
        if fill == "random":
            if "n_cells" not in reg:
                sys.exit(f"[error] region '{reg.get('name','?')}': "
                         f"random fill needs \"n_cells\"")
            return random_points_radial(cx, cy, r_in, r_out, int(reg["n_cells"]), rng)
        return hex_points_radial(cx, cy, r_in, r_out, s)

    if shape != "rect":
        sys.exit(f"[error] region '{reg.get('name','?')}': unknown shape '{shape}' "
                 f"(expected rect, disk, or annulus)")

    x0, x1, y0, y1 = region_abs(reg)
    x0 += GAP/2; x1 -= GAP/2; y0 += GAP/2; y1 -= GAP/2   # vacuum gap between regions
    if fill == "random":
        if "n_cells" not in reg:
            sys.exit(f"[error] region '{reg.get('name','?')}': random fill needs \"n_cells\"")
        return random_points_rect(x0, x1, y0, y1, int(reg["n_cells"]), rng)
    return hex_points(x0, x1, y0, y1, s)


def check_against_xml(xml_path, used_types):
    """Cross-check every type name in cells.csv against <cell_definition> names.

    PhysiCell drops cells whose type name has no matching definition and does
    not report it. This is what silently removed the eighth skin type. The
    check is cheap; run it every time.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        sys.exit(f"[error] could not read XML '{xml_path}': {e}")
    defs = root.find("cell_definitions")
    if defs is None:
        sys.exit(f"[error] '{xml_path}' has no <cell_definitions> block")
    defined = {cd.get("name") for cd in defs.findall("cell_definition")}
    missing = sorted(used_types - defined)
    if missing:
        print(f"\n[error] these cells.csv types have no <cell_definition> and would be "
              f"DROPPED SILENTLY:", file=sys.stderr)
        for t in missing:
            print(f"          {t}", file=sys.stderr)
        print(f"        XML defines: {sorted(defined)}", file=sys.stderr)
        sys.exit(1)
    unused = sorted(defined - used_types)
    print(f"[ok] all {len(used_types)} type names matched in {xml_path}")
    if unused:
        print(f"     (defined but unplaced, which may be intentional: {unused})")


def main(check_xml=None):
    rng = np.random.default_rng(SEED)
    rows = []
    for reg in get_regions():
        if "composition" not in reg:
            sys.exit(f"[error] region '{reg.get('name','?')}' has no \"composition\"")
        pts = region_points(reg, rng)
        for (x, y), t in zip(pts, assign_types(len(pts), reg["composition"], rng)):
            rows.append((x, y, DOMAIN.get("z", 0.0), t))

    if not rows:
        sys.exit("[error] no cells placed -- check region bounds and spacing")

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x", "y", "z", "type"]); w.writerows(rows)
    print(f"Wrote {len(rows)} cells -> {OUTPUT_CSV}")
    print("Counts by type:")
    for t, c in sorted(Counter(r[3] for r in rows).items()):
        print(f"  {t:20s} {c}")

    if check_xml:
        check_against_xml(check_xml, {r[3] for r in rows})

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d = DOMAIN; fig, ax = plt.subplots(figsize=(6, 6))
        for t in sorted(set(r[3] for r in rows)):
            xs = [r[0] for r in rows if r[3] == t]; ys = [r[1] for r in rows if r[3] == t]
            ax.scatter(xs, ys, s=8, label=t, color=get_color(t))
        ax.set_xlim(d["x_min"], d["x_max"]); ax.set_ylim(d["y_min"], d["y_max"])
        ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper right"); ax.set_title("cells.csv preview")
        png = OUTPUT_CSV.rsplit(".", 1)[0] + "_preview.png"
        fig.savefig(png, dpi=120, bbox_inches="tight"); print(f"Preview -> {png}")
    except Exception as e:
        print("(preview skipped:", e, ")")


def configure(cfg):
    """Bind config to module globals. Called from __main__, and available to
    anything that wants to import this module rather than shell out to it."""
    global CELL_TYPES, GROUPS, DOMAIN, SPACING, GAP, SEED, BANDS, REGIONS, OUTPUT_CSV
    CELL_TYPES = cfg.get("cell_types", {})
    GROUPS     = cfg.get("groups", {})
    DOMAIN     = cfg["domain"]
    SPACING    = cfg.get("spacing", 18.0)
    GAP        = cfg.get("gap", 30.0)
    SEED       = cfg.get("seed", 0)
    BANDS      = cfg.get("bands", [])
    REGIONS    = cfg.get("regions", [])
    OUTPUT_CSV = cfg["output_csv"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a PhysiCell cells.csv from a JSON config.")
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--check-xml", metavar="PATH",
                    help="verify every type name has a <cell_definition> in this XML")
    ap.add_argument("--seed", type=int, help="override the config seed")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.seed is not None: cfg["seed"] = a.seed
    configure(cfg)
    main(check_xml=a.check_xml)
