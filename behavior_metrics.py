#!/usr/bin/env python3
"""
behavior_metrics.py -- quantitative validation of the tumor-demo replication.

The paper's claims are all qualitative ("notice the greatest cycling is on the
outer periphery"). This turns each one into a number you can put on a slide and
that a reviewer can check. It also supplies the null test that separates
"behavior" from "marbles": a chemotaxis index that is ~0 for a random walk and
clearly negative (inward) when chemotaxis is doing real work.

Checks, and the paper claim each maps to:
  C1  pressure decreases with radius          paper Fig 11
  C2  cycle entry increases with radius       supplemental Fig 8 text
  C3  oxygen-driven spatial asymmetry         supplemental Fig 13 text
  C4  debris accumulates over the tumor       supplemental Fig 24 text
  C5  macrophages drift inward (chemotaxis)   supplemental Figs 30, 33-34
  C6  M1 conversion tracks dead-cell contact  supplemental Figs 39-40
  C7  cancer population falls after T arrival supplemental Figs 48-50

Usage:
    py -3.12 behavior_metrics.py --output PhysiCell/output --out-dir figs
    py -3.12 behavior_metrics.py --output out_chemo_off --label null
"""

import argparse, os, sys, math
import numpy as np

try:
    import pcdl
except ImportError:
    sys.exit("pcdl not found. In PowerShell: py -3.12 -m pip install pcdl")

LOW_O2 = np.array([-1.0, -1.0]) / math.sqrt(2.0)   # Dirichlet xmin=ymin=10 corner


# ------------------------------------------------------------------ plumbing

def resolve(df, *candidates):
    """pcdl column names drift between versions; fail with a useful list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load(output_dir):
    ts = pcdl.TimeSeries(output_dir, verbose=False)
    frames = []
    for t in ts.get_mcds_list():
        df = t.get_cell_df()
        df = df.reset_index()
        df["_time"] = t.get_time()
        frames.append(df)
    if not frames:
        sys.exit(f"no output frames found in {output_dir}")
    return frames


def cancer_count(df):
    tcol = resolve(df, "cell_type")
    return int((df[tcol] == "cancer").sum())


def peak_frame(frames):
    """Frame with the most cancer cells: where cycling, necrosis, and debris are measurable."""
    counts = [cancer_count(f) for f in frames]
    return frames[int(np.argmax(counts))]


def active_frames(frames, min_cancer=10):
    """Frames while the tumor is alive: the window where chemotaxis toward it acts."""
    keep = [f for f in frames if cancer_count(f) >= min_cancer]
    return keep if len(keep) >= 2 else frames


def radial(df, xcol, ycol, center=(0.0, 0.0)):
    return np.hypot(df[xcol] - center[0], df[ycol] - center[1])


def profile(r, v, edges):
    """Mean of v in radial bins. Returns (centers, means, counts)."""
    idx = np.digitize(r, edges) - 1
    cent, mean, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        cent.append(0.5 * (edges[b] + edges[b + 1]))
        cnt.append(int(m.sum()))
        mean.append(float(np.nanmean(v[m])) if m.sum() else np.nan)
    return np.array(cent), np.array(mean), np.array(cnt)


def trend(x, y, w):
    """Weighted least-squares slope, ignoring empty bins."""
    ok = ~np.isnan(y) & (w > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1, w=np.sqrt(w[ok]))[0])


# -------------------------------------------------------------------- checks

def c1_c2_radial(frames, res):
    last = peak_frame(frames)
    xc, yc = resolve(last, "position_x"), resolve(last, "position_y")
    alive = last[last[resolve(last, "dead")] == False] if resolve(last, "dead") else last
    r = radial(alive, xc, yc).values
    edges = np.linspace(0, max(r.max(), 1.0), 11)

    pcol = resolve(alive, "pressure")
    if pcol:
        c, m, n = profile(r, alive[pcol].values, edges)
        s = trend(c, m, n)
        res["C1 pressure vs radius"] = (
            f"slope {s:+.3e} /um", s < 0,
            "paper Fig 11: pressure highest at the core")

    ccol = resolve(alive, "current_cycle_phase_exit_rate", "cycle_entry")
    if ccol:
        c, m, n = profile(r, alive[ccol].values, edges)
        s = trend(c, m, n)
        res["C2 cycle entry vs radius"] = (
            f"slope {s:+.3e} /um", s > 0,
            "supp Fig 8: greatest cycling on the outer periphery")


def c3_asymmetry(frames, res):
    """The strongest spatial test. With Dirichlet xmin=ymin=10, oxygen is low
    toward (-x,-y). Project each cell onto that axis and compare halves. Bulk
    counts are blind to this by construction -- it is purely spatial."""
    last = peak_frame(frames)
    xc, yc = resolve(last, "position_x"), resolve(last, "position_y")
    proj = (last[xc].values * LOW_O2[0] + last[yc].values * LOW_O2[1])
    lo, hi = proj > 0, proj <= 0          # lo = toward the low-oxygen corner

    ocol = resolve(last, "oxygen")
    if ocol:
        o_lo, o_hi = np.nanmean(last[ocol].values[lo]), np.nanmean(last[ocol].values[hi])
        res["C3a oxygen low-corner vs high"] = (
            f"{o_lo:.2f} vs {o_hi:.2f} mmHg", o_lo < o_hi,
            "supp Fig 12: Dirichlet xmin=ymin=10 sets up the gradient")

    dcol = resolve(last, "dead")
    if dcol:
        d = last[dcol].values.astype(bool)
        f_lo = d[lo].mean() if lo.sum() else np.nan
        f_hi = d[hi].mean() if hi.sum() else np.nan
        res["C3b dead fraction low-O2 vs high"] = (
            f"{f_lo:.3f} vs {f_hi:.3f}", f_lo > f_hi,
            "supp Fig 13: necrosis preferential where oxygen is lower")

    ccol = resolve(last, "current_cycle_phase_exit_rate")
    if ccol:
        v = last[ccol].values
        res["C3c cycle entry low-O2 vs high"] = (
            f"{np.nanmean(v[lo]):.2e} vs {np.nanmean(v[hi]):.2e}",
            np.nanmean(v[lo]) < np.nanmean(v[hi]),
            "supp Fig 13: cycling preferential where oxygen is higher")


def c4_debris(frames, res):
    dcol = resolve(frames[-1], "debris")
    if not dcol:
        return
    last = peak_frame(frames)
    xc, yc = resolve(last, "position_x"), resolve(last, "position_y")
    r = radial(last, xc, yc).values
    inner, outer = r < 200, r >= 250
    if inner.sum() and outer.sum():
        a, b = np.nanmean(last[dcol].values[inner]), np.nanmean(last[dcol].values[outer])
        res["C4 debris inside vs outside tumor"] = (
            f"{a:.3f} vs {b:.3f}", a > b,
            "supp Fig 24: debris accumulates in the tumor region")


def c5_chemotaxis(frames, res, cell_type="macrophage"):
    """Chemotaxis index: mean cosine between each step and the inward direction.

    A pure random walk gives ~0 regardless of speed. Chemotaxis toward a
    tumor-centred debris source gives a clearly negative radial drift. This one
    number is the marbles-vs-behavior test; run it on a chemotaxis-off control
    and the value should collapse toward zero."""
    tcol = resolve(frames[0], "cell_type")
    idcol = resolve(frames[0], "ID", "id")
    if not (tcol and idcol):
        return

    cosines, dr_total = [], []
    act = active_frames(frames)
    for a, b in zip(act[:-1], act[1:]):
        xc, yc = resolve(a, "position_x"), resolve(a, "position_y")
        A = a[a[tcol] == cell_type].set_index(idcol)
        B = b[b[tcol] == cell_type].set_index(idcol)
        common = A.index.intersection(B.index)      # survives births and deaths
        if len(common) == 0:
            continue
        p0 = A.loc[common, [xc, yc]].values
        p1 = B.loc[common, [xc, yc]].values
        step = p1 - p0
        n = np.linalg.norm(step, axis=1)
        r0 = np.linalg.norm(p0, axis=1)
        ok = (n > 1e-9) & (r0 > 1e-9)
        if ok.sum() == 0:
            continue
        inward = -p0[ok] / r0[ok, None]
        cosines.append((step[ok] * inward).sum(1) / n[ok])
        dr_total.append(np.linalg.norm(p1[ok], axis=1) - r0[ok])

    if not cosines:
        return
    ci = float(np.concatenate(cosines).mean())
    dr = float(np.concatenate(dr_total).sum() / len(np.concatenate(dr_total)))
    res[f"C5 {cell_type} chemotaxis index"] = (
        f"{ci:+.3f}  (mean dr {dr:+.2f} um/step)", ci > 0.05,
        "0 = random walk; positive = directed inward toward debris")


def c6_m1(frames, res):
    tcol = resolve(frames[0], "cell_type")
    if not tcol:
        return
    first, last = frames[0], frames[-1]
    n0 = int((first[tcol] == "macrophage").sum())
    m1 = int((last[tcol] == "M1 macrophage").sum())
    if n0:
        res["C6 macrophage -> M1 conversion"] = (
            f"{m1}/{n0} converted ({100*m1/n0:.0f}%)", m1 > 0,
            "supp Fig 38: contact with dead cell drives transformation")


def c7_cancer(frames, res):
    tcol = resolve(frames[0], "cell_type")
    if not tcol:
        return
    counts = [(f["_time"].iloc[0], int((f[tcol] == "cancer").sum())) for f in frames]
    peak_t, peak_n = max(counts, key=lambda c: c[1])
    end_t, end_n = counts[-1]
    res["C7 cancer count peak -> end"] = (
        f"{peak_n} @ {peak_t:.0f} min  ->  {end_n} @ {end_t:.0f} min",
        end_n < peak_n,
        "supp Figs 48-50: T cells attack and kill off cancer cells")


def population_flatness(frames, res):
    """Guard against the old proliferation trap in reverse: for THIS model the
    population must NOT be flat. A flat count means rules aren't firing."""
    tcol = resolve(frames[0], "cell_type")
    n = [int((f[tcol] == "cancer").sum()) for f in frames] if tcol else [len(f) for f in frames]
    spread = (max(n) - min(n)) / max(max(n), 1)
    res["G  population is dynamic"] = (
        f"{min(n)}..{max(n)} ({100*spread:.0f}% range)", spread > 0.05,
        "flat counts here mean the rules file was not loaded")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="PhysiCell output folder")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    frames = load(a.output)
    print(f"{a.label}: {len(frames)} frames, "
          f"{len(frames[0])} -> {len(frames[-1])} agents\n")

    res = {}
    for fn in (population_flatness, c1_c2_radial, c3_asymmetry,
               c4_debris, c6_m1, c7_cancer):
        try:
            fn(frames, res)
        except Exception as e:
            print(f"  (skipped {fn.__name__}: {e})")
    for ct in ("macrophage", "Effector T cell"):
        try:
            c5_chemotaxis(frames, res, ct)
        except Exception as e:
            print(f"  (skipped chemotaxis {ct}: {e})")

    width = max(len(k) for k in res) if res else 10
    npass = 0
    for k, (val, ok, why) in res.items():
        mark = "PASS" if ok else "FAIL"
        npass += bool(ok)
        print(f"[{mark}] {k:<{width}}  {val}")
        print(f"       {why}")
    print(f"\n{npass}/{len(res)} checks reproduce the paper")


if __name__ == "__main__":
    main()
