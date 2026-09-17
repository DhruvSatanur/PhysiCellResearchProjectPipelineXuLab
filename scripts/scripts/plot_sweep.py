#!/usr/bin/env python3
"""
plot_sweep.py -- figures for the sweep analysis.

This is the stage the pipeline was missing: sweep_driver.py reduces every run to
scalars and writes sweep_results.csv + discriminability.csv, but nothing turned
those into anything you can put in front of a reader.

Three figures, each answering one question:

  fig1_discriminability  Which metrics separate parameter sets relative to seed
                         noise? This is the project's central claim as a picture.
                         Bulk and spatial are colored differently; if the spatial
                         bars sit above the bulk bars, spatial metrics carry
                         information a calibration could use that bulk does not.

  fig2_bulk_vs_spatial   The same data collapsed to a two-group comparison, with
                         every metric shown as a point so a single outlier cannot
                         masquerade as a trend. Use this one in a talk; use fig1
                         when someone asks which metric specifically.

  fig3_response_curves   For the top metrics, value against each swept parameter,
                         one point per seed plus the per-set mean. This is how you
                         check that a high F is a real monotone response and not
                         one parameter set landing somewhere odd.

It does NOT recompute the F ratios. It reads discriminability.csv so the figure
can never disagree with the table sweep_driver printed. Run aggregate first.

Usage:
    py -3.12 plot_sweep.py                          # uses sweep_config.json
    py -3.12 plot_sweep.py --config other.json
    py -3.12 plot_sweep.py --out-dir figs --top 8
    py -3.12 plot_sweep.py --no-pdf                 # PNG only, faster
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Columns sweep_driver writes that describe the run rather than measure it.
# Anything in sweep_results.csv that is not one of these and not bulk_/spat_
# prefixed is treated as a swept parameter.
BOOKKEEPING = {
    "run_id", "param_set", "seed", "status", "returncode",
    "wall_seconds", "n_frames", "t_end",
}

BULK_COLOR = "#8a8f98"      # grey: the conventional baseline
SPAT_COLOR = "#2f6fb1"      # blue: the thing under test


def style():
    """Defaults that survive being dropped into a poster or a slide."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def pretty(col):
    """bulk_mean_density -> mean density. Keeps axis labels readable without
    losing which family the metric belongs to (that is carried by color)."""
    for p in ("bulk_", "spat_"):
        if col.startswith(p):
            return col[len(p):].replace("_", " ")
    return col.replace("_", " ")


def load(work, out_dir):
    """Read both tables, or explain exactly which step has not been run."""
    res_p = work / "sweep_results.csv"
    dis_p = work / "discriminability.csv"

    if not res_p.exists():
        sys.exit(f"{res_p} not found.\n"
                 f"  Run:  py -3.12 sweep_driver.py aggregate")
    res = pd.read_csv(res_p)

    if not dis_p.exists():
        sys.exit(f"{dis_p} not found, though {res_p.name} exists.\n"
                 f"  aggregate skips discriminability when any run is missing or\n"
                 f"  failed. Re-run the incomplete run_ids it listed, then aggregate\n"
                 f"  again. Do not plot a partial sweep as if it were complete.")
    dis = pd.read_csv(dis_p)

    if dis.empty:
        sys.exit("discriminability.csv is empty: no metric varied enough to score.")

    out_dir.mkdir(parents=True, exist_ok=True)
    return res, dis


def swept_params(res):
    """Discover which columns are swept parameters rather than metrics.

    Derived from the file instead of hardcoded so this keeps working when the
    sweep design changes, which it will.
    """
    cols = [c for c in res.columns
            if c not in BOOKKEEPING
            and not c.startswith(("bulk_", "spat_"))]
    # A parameter that took one value tells you nothing and would plot as a
    # single vertical stripe.
    return [c for c in cols if res[c].nunique() > 1]


def save(fig, out_dir, name, want_pdf):
    """PNG for the README and Slack, PDF for anything that gets printed."""
    paths = []
    p = out_dir / f"{name}.png"
    fig.savefig(p)
    paths.append(p)
    if want_pdf:
        p = out_dir / f"{name}.pdf"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- figure 1

def fig_discriminability(dis, out_dir, top, want_pdf):
    """Horizontal bars, one per metric, sorted by F, colored by family.

    Log x-axis: F ratios routinely span two or three orders of magnitude, and on
    a linear axis the top metric flattens everything below it into a stub.
    """
    d = dis.sort_values("F", ascending=True)
    if top and len(d) > top:
        d = d.tail(top)          # tail because ascending, so we keep the largest

    h = max(2.6, 0.34 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, h))

    colors = [SPAT_COLOR if k == "spatial" else BULK_COLOR for k in d["kind"]]
    y = np.arange(len(d))
    ax.barh(y, d["F"].values, color=colors, height=0.72)

    ax.set_yticks(y)
    ax.set_yticklabels([pretty(m) for m in d["metric"]])
    ax.set_xscale("log")
    ax.set_xlabel("F  =  between-parameter-set variance / within-set (seed) variance")
    ax.set_title("Which metrics separate parameter sets above seed noise")
    ax.grid(axis="y", visible=False)

    # F = 1 is the reference: a metric scoring at or below it varies as much
    # between seeds of one setting as it does between settings.
    ax.axvline(1.0, color="#c0392b", lw=1.0, ls="--", zorder=0)
    ax.text(1.0, len(d) - 0.3, "  F = 1: no better than seed noise",
            color="#c0392b", fontsize=8, va="top")

    handles = [plt.Rectangle((0, 0), 1, 1, color=SPAT_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=BULK_COLOR)]
    ax.legend(handles, ["spatial (single-cell)", "bulk (conventional)"],
              loc="lower right")

    return save(fig, out_dir, "fig1_discriminability", want_pdf)


# ---------------------------------------------------------------- figure 2

def fig_bulk_vs_spatial(dis, out_dir, want_pdf):
    """Two groups, every metric plotted, medians marked.

    A box plot alone would hide how few metrics are in each group. The jittered
    points are the honest version: with six metrics a side, the reader should be
    able to count them.
    """
    groups = ["bulk", "spatial"]
    data = [dis.loc[dis["kind"] == g, "F"].values for g in groups]

    if any(len(x) == 0 for x in data):
        present = dis["kind"].unique().tolist()
        print(f"  skipping fig2: only {present} metrics scored, nothing to compare")
        return []

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    rng = np.random.default_rng(0)      # fixed: the figure must not move between runs

    for i, (g, vals) in enumerate(zip(groups, data)):
        color = SPAT_COLOR if g == "spatial" else BULK_COLOR
        x = i + rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(x, vals, s=34, color=color, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.6)
        med = float(np.median(vals))
        ax.hlines(med, i - 0.26, i + 0.26, color=color, lw=2.4, zorder=4)
        ax.text(i + 0.32, med, f"median {med:.1f}\nn = {len(vals)}",
                fontsize=8, va="center", color=color)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["bulk\n(conventional)", "spatial\n(single-cell)"])
    ax.set_yscale("log")
    ax.set_ylabel("F  (higher = separates parameter sets better)")
    ax.set_title("Bulk vs spatial metrics")
    ax.set_xlim(-0.5, 1.75)
    ax.axhline(1.0, color="#c0392b", lw=1.0, ls="--", zorder=0)
    ax.grid(axis="x", visible=False)

    return save(fig, out_dir, "fig2_bulk_vs_spatial", want_pdf)


# ---------------------------------------------------------------- figure 3

def fig_response_curves(res, dis, params, out_dir, top, want_pdf):
    """Metric value against each swept parameter: rows = metrics, cols = params.

    Seeds are plotted individually rather than as error bars. With three seeds an
    error bar implies more than you have, and seeing the three points is what
    tells you whether a clean-looking mean is actually reproducible.
    """
    if not params:
        print("  skipping fig3: no parameter varied across runs")
        return []

    metrics = dis.sort_values("F", ascending=False)["metric"].head(top).tolist()
    metrics = [m for m in metrics if m in res.columns]
    if not metrics:
        print("  skipping fig3: no scored metric found in sweep_results.csv")
        return []

    nr, nc = len(metrics), len(params)
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.35 * nr),
                             squeeze=False, sharex="col")

    for r, metric in enumerate(metrics):
        kind = dis.loc[dis["metric"] == metric, "kind"].iloc[0]
        fval = dis.loc[dis["metric"] == metric, "F"].iloc[0]
        color = SPAT_COLOR if kind == "spatial" else BULK_COLOR

        for c, param in enumerate(params):
            ax = axes[r][c]
            sub = res[[param, metric]].dropna()

            # Every seed and every other parameter's level shows up here, so the
            # vertical spread at one x is the combined noise, not just seeds.
            ax.scatter(sub[param], sub[metric], s=20, color=color,
                       alpha=0.45, zorder=2, edgecolor="none")
            g = sub.groupby(param)[metric].mean()
            ax.plot(g.index, g.values, color=color, lw=1.8, marker="o",
                    ms=4.5, zorder=3)

            if c == 0:
                ax.set_ylabel(f"{pretty(metric)}\nF = {fval:.1f}", fontsize=9)
            if r == nr - 1:
                ax.set_xlabel(param.replace("_", " "), fontsize=9)
            ax.tick_params(labelsize=8)

    fig.suptitle("Metric response to each swept parameter "
                 "(faint points = individual runs, line = mean per level)",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    return save(fig, out_dir, "fig3_response_curves", want_pdf)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="sweep_config.json",
                    help="same config sweep_driver.py uses; supplies work_root")
    ap.add_argument("--work-root", default=None,
                    help="override work_root instead of reading the config")
    ap.add_argument("--out-dir", default=None,
                    help="where figures go (default: <work_root>/figs)")
    ap.add_argument("--top", type=int, default=12,
                    help="how many metrics to show (default 12)")
    ap.add_argument("--no-pdf", action="store_true",
                    help="skip the vector copies")
    a = ap.parse_args()

    if a.work_root:
        work = Path(a.work_root)
    else:
        if not Path(a.config).exists():
            sys.exit(f"{a.config} not found. Pass --config or --work-root.")
        with open(a.config) as f:
            work = Path(json.load(f)["work_root"])

    out_dir = Path(a.out_dir) if a.out_dir else work / "figs"
    style()

    res, dis = load(work, out_dir)
    params = swept_params(res)

    n_sets = res["param_set"].nunique() if "param_set" in res else "?"
    print(f"{len(res)} runs, {n_sets} parameter sets, "
          f"{len(dis)} scored metrics, swept: {', '.join(params) or 'none'}\n")

    written = []
    written += fig_discriminability(dis, out_dir, a.top, not a.no_pdf)
    written += fig_bulk_vs_spatial(dis, out_dir, not a.no_pdf)
    written += fig_response_curves(res, dis, params, out_dir, min(a.top, 4),
                                   not a.no_pdf)

    for p in written:
        print(f"wrote {p}")

    # The headline number, printed so it ends up in the terminal log next to the
    # figures rather than only inside them.
    med = dis.groupby("kind")["F"].median()
    if {"bulk", "spatial"} <= set(med.index):
        print(f"\nmedian F: bulk {med['bulk']:.2f}, spatial {med['spatial']:.2f}")
        if med["spatial"] <= med["bulk"]:
            print("Spatial does not beat bulk on this sweep. That is a result, "
                  "not a bug. Report it.")


if __name__ == "__main__":
    main()
