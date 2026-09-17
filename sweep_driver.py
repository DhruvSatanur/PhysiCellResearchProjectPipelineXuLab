#!/usr/bin/env python3
"""
sweep_driver.py -- parameter sweeps for the spatial-metrics project.

Three subcommands:

    plan       expand the design into runs/manifest.csv (one row per run)
    run N      execute run N from the manifest, write its metrics row
    aggregate  collect all rows, REPORT WHAT IS MISSING, write sweep_results.csv

Each run gets its own config XML and its own output directory. That isolation
is not optional: PhysiCell does not clear output/ between runs, so concurrent
array tasks sharing a folder would read each other's frames and silently
produce results that look fine.

Every run records BOTH bulk and spatial metrics, prefixed bulk_ and spat_, so
the discriminability comparison later is just a column selection.

Local pilot:
    py -3.12 sweep_driver.py plan
    py -3.12 sweep_driver.py run 0          # check one works end to end
    for /L %i in (0,1,14) do py -3.12 sweep_driver.py run %i
    py -3.12 sweep_driver.py aggregate

Zaratan: see sweep_array.sbatch.
"""

import argparse, csv, itertools, json, os, platform, shutil, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_distance_metrics import _normalize, local_niche, displacement_by_type, survival_summary


# ------------------------------------------------------------------ planning

def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def expand(cfg):
    """Full factorial over parameters, crossed with seeds."""
    params = {k: v for k, v in cfg["parameters"].items() if not k.startswith("_")}
    names = list(params)
    grids = [params[n]["values"] for n in names]
    runs, rid = [], 0
    for combo in itertools.product(*grids):
        for seed in cfg["seeds"]:
            row = {"run_id": rid, "seed": seed}
            row.update({n: v for n, v in zip(names, combo)})
            # param_set groups the seeds that share one parameter combination.
            # The whole discriminability analysis depends on this column.
            row["param_set"] = "|".join(f"{n}={v}" for n, v in zip(names, combo))
            runs.append(row)
            rid += 1
    return runs, names


def cmd_plan(cfg, args):
    runs, names = expand(cfg)
    work = Path(cfg["work_root"])
    (work / "runs").mkdir(parents=True, exist_ok=True)
    man = work / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "param_set", "seed"] + names)
        w.writeheader()
        w.writerows(runs)
    n_sets = len(runs) // max(len(cfg["seeds"]), 1)
    print(f"planned {len(runs)} runs = {n_sets} parameter sets x {len(cfg['seeds'])} seeds")
    print(f"manifest -> {man}")
    print(f"\nswept: {', '.join(names)}")
    print(f"\nSLURM array range:  0-{len(runs)-1}")
    print("\nEstimate total time as (single-run wall time) x runs / parallel tasks.")
    print("If you have not timed a single run yet, do that before scaling up.")


# ------------------------------------------------------------------ one run

def patch_xml(base, out, cfg, row):
    tree = ET.parse(base)
    root = tree.getroot()

    def setpath(xpath, value, what):
        node = root.find(xpath)
        if node is None:
            sys.exit(f"[run {row['run_id']}] xpath not found for {what}: {xpath}\n"
                     f"  check it against {base}")
        node.text = str(value)

    for name, spec in cfg["parameters"].items():
        if name.startswith("_"):
            continue
        setpath(spec["xpath"], row[name], name)

    setpath(cfg["seed_xpath"], row["seed"], "seed")
    setpath(cfg["output_folder_xpath"], row["_output"], "output folder")
    if cfg.get("max_time_override"):
        setpath(cfg["max_time_xpath"], cfg["max_time_override"], "max_time")

    tree.write(out, encoding="UTF-8", xml_declaration=True)


def region_of(x, y, an):
    """Assign each cell to a named region. Returns an array of names."""
    if an.get("regions_radial"):
        r = an["regions_radial"]
        cx, cy = r["center"]
        d = np.hypot(x - cx, y - cy)
        idx = np.clip(np.digitize(d, r["edges"]) - 1, 0, len(r["names"]) - 1)
        return np.array(r["names"])[idx]
    if an.get("regions_bands"):
        b = an["regions_bands"]
        idx = np.clip(np.digitize(y, b["edges"]) - 1, 0, len(b["names"]) - 1)
        return np.array(b["names"])[idx]
    return np.full(len(x), "all")


def metrics(output_dir, cfg):
    """Compute bulk and spatial metrics for one completed run."""
    import pcdl
    an = cfg["analysis"]
    ts = pcdl.TimeSeries(str(output_dir), verbose=False)
    mcds = ts.get_mcds_list()
    if len(mcds) < 2:
        raise RuntimeError(f"only {len(mcds)} frames; run did not complete")

    frames = [(m.get_time(), m.get_cell_df().reset_index()) for m in mcds]
    t0, f0 = frames[0]
    tN, fN = frames[-1]
    m = {"n_frames": len(frames), "t_end": tN}

    # ---- BULK: what a conventional calibration would use
    d0 = _normalize(f0, exclude_dead=an["exclude_dead"])
    dN = _normalize(fN, exclude_dead=an["exclude_dead"])
    m["bulk_n_start"] = len(d0)
    m["bulk_n_end"] = len(dN)
    m["bulk_growth_ratio"] = len(dN) / max(len(d0), 1)
    for t in sorted(set(d0["type"]) | set(dN["type"])):
        m[f"bulk_n_{t}"] = int((dN["type"] == t).sum())
        m[f"bulk_frac_{t}"] = float((dN["type"] == t).mean())
    niche = local_niche(fN, radius=an["niche_radius"], exclude_dead=an["exclude_dead"])
    m["bulk_mean_density"] = float(niche["local_density"].mean())

    # ---- SPATIAL: single-cell, the thing the project claims adds information
    disp = displacement_by_type(f0, fN)
    surv = survival_summary(f0, fN)
    m["spat_frac_tracked"] = surv["frac_tracked"]
    m["spat_mean_disp"] = float(disp["displacement"].mean())
    m["spat_median_disp"] = float(disp["displacement"].median())
    m["spat_std_disp"] = float(disp["displacement"].std())

    for ft in an["focal_types"]:
        sub = disp[disp["type_start"] == ft]
        if len(sub) == 0:
            continue
        m[f"spat_disp_{ft}"] = float(sub["displacement"].mean())

        # net radial drift: negative means moved inward. A random walk gives ~0
        # regardless of speed, so this separates directed motion from fast noise.
        a = _normalize(f0).set_index("ID")
        b = _normalize(fN).set_index("ID")
        ids = sub["ID"].values
        r0 = np.hypot(a.loc[ids, "x"], a.loc[ids, "y"]).values
        r1 = np.hypot(b.loc[ids, "x"], b.loc[ids, "y"]).values
        m[f"spat_radial_drift_{ft}"] = float(np.mean(r1 - r0))
        m[f"spat_transformed_frac_{ft}"] = float(sub["transformed"].mean())

    # residence time per region, and the niche composition experienced
    focal = set(an["focal_types"])
    res, comp_acc, n_acc = {}, {}, 0
    for t, f in frames:
        d = _normalize(f, exclude_dead=an["exclude_dead"])
        sel = d["type"].isin(focal) if focal else np.ones(len(d), bool)
        if sel.sum() == 0:
            continue
        regs = region_of(d.loc[sel, "x"].values, d.loc[sel, "y"].values, an)
        for r in np.unique(regs):
            res[r] = res.get(r, 0) + int((regs == r).sum())
        n_acc += int(sel.sum())
    for r, c in res.items():
        m[f"spat_residence_frac_{r}"] = c / max(n_acc, 1)

    nsel = niche["type"].isin(focal) if focal else np.ones(len(niche), bool)
    if nsel.sum():
        for c in [c for c in niche.columns if c.startswith("frac_")]:
            m[f"spat_niche_{c[5:]}"] = float(niche.loc[nsel, c].mean())
        m["spat_density_experienced"] = float(niche.loc[nsel, "local_density"].mean())

    return m


def cmd_run(cfg, args):
    work = Path(cfg["work_root"])
    man = pd.read_csv(work / "manifest.csv")
    if args.n not in man["run_id"].values:
        sys.exit(f"run_id {args.n} not in manifest (0-{man['run_id'].max()})")
    row = man[man["run_id"] == args.n].iloc[0].to_dict()

    rdir = (work / "runs" / f"run{args.n:05d}").resolve()
    odir = rdir / "output"
    if rdir.exists():
        shutil.rmtree(rdir)          # never inherit a stale output folder
    odir.mkdir(parents=True)

    root = Path(cfg["physicell_root"]).resolve()
    row["_output"] = str(odir)
    xml = rdir / "config.xml"
    patch_xml(root / cfg["base_config"], xml, cfg, row)

    exe = cfg["executable_windows"] if platform.system() == "Windows" else cfg["executable"]
    exe_path = root / exe
    if not exe_path.exists():
        sys.exit(f"executable not found: {exe_path}")

    env = os.environ.copy()
    if cfg.get("dll_path"):
        env["PATH"] = cfg["dll_path"] + os.pathsep + env.get("PATH", "")

    t0 = time.time()
    log = rdir / "stdout.log"
    with open(log, "w") as lf:
        proc = subprocess.run([str(exe_path), str(xml)], cwd=root,
                              stdout=lf, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0

    rec = {k: v for k, v in row.items() if not k.startswith("_")}
    rec["wall_seconds"] = round(wall, 1)
    rec["returncode"] = proc.returncode

    if proc.returncode != 0:
        rec["status"] = "sim_failed"
        print(f"run {args.n}: SIM FAILED rc={proc.returncode}, see {log}", file=sys.stderr)
    else:
        try:
            rec.update(metrics(odir, cfg))
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = f"analysis_failed: {type(e).__name__}: {e}"
            print(f"run {args.n}: ANALYSIS FAILED: {e}", file=sys.stderr)

    # one file per run: no concurrent-write races across array tasks
    with open(rdir / "metrics.json", "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"run {args.n}: {rec['status']} in {wall:.1f}s")
    if rec["status"] != "ok":
        sys.exit(1)


# ------------------------------------------------------------------ aggregate

def cmd_aggregate(cfg, args):
    work = Path(cfg["work_root"])
    man = pd.read_csv(work / "manifest.csv")
    rows, missing, failed = [], [], []

    for rid in man["run_id"]:
        p = work / "runs" / f"run{rid:05d}" / "metrics.json"
        if not p.exists():
            missing.append(rid)
            continue
        r = json.load(open(p))
        (rows if r.get("status") == "ok" else failed).append(r)
        if r.get("status") == "ok":
            pass

    print(f"planned {len(man)} | ok {len(rows)} | failed {len(failed)} | never ran {len(missing)}")
    if missing:
        print(f"  MISSING run_ids: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        print("  A silently missing array task looks identical to a smaller sweep.")
        print("  Re-run these before trusting any aggregate.")
    for r in failed:
        print(f"  run {r['run_id']}: {r['status']}")

    if not rows:
        sys.exit("no successful runs to aggregate")

    df = pd.DataFrame(rows)
    out = work / "sweep_results.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows x {len(df.columns)} cols)")

    if "wall_seconds" in df:
        print(f"wall time per run: median {df.wall_seconds.median():.1f}s, "
              f"max {df.wall_seconds.max():.1f}s")

    if len(missing) or len(failed):
        print("\nNOT computing discriminability: the sweep is incomplete.")
        return
    discriminability(df, work)


def discriminability(df, work):
    """Between-parameter-set variance over within-set (seed) variance.

    This is the project's central claim reduced to one table. A metric scores
    high when parameter sets separate cleanly relative to seed noise, i.e. when
    the metric carries information a calibration could actually use.

    Read it as a comparison BETWEEN metrics on the same runs, not as a p-value.
    """
    metric_cols = [c for c in df.columns
                   if (c.startswith("bulk_") or c.startswith("spat_"))
                   and pd.api.types.is_numeric_dtype(df[c])]
    n_sets = df["param_set"].nunique()
    if n_sets < 2:
        print("\nonly one parameter set -- nothing to discriminate")
        return

    out = []
    for c in metric_cols:
        v = df[[c, "param_set"]].dropna()
        if v[c].std() == 0 or len(v) < 4:
            continue
        grand = v[c].mean()
        g = v.groupby("param_set")[c]
        between = sum(len(x) * (x.mean() - grand) ** 2 for _, x in g) / max(n_sets - 1, 1)
        within = sum(((x - x.mean()) ** 2).sum() for _, x in g) / max(len(v) - n_sets, 1)
        if within <= 0:
            continue
        out.append({"metric": c,
                    "kind": "bulk" if c.startswith("bulk_") else "spatial",
                    "F": between / within})

    if not out:
        print("\nno metrics varied enough to score")
        return

    res = pd.DataFrame(out).sort_values("F", ascending=False)
    res.to_csv(work / "discriminability.csv", index=False)
    print(f"\ndiscriminability (between-set var / within-set var), "
          f"{n_sets} sets:\n")
    print(res.to_string(index=False, float_format=lambda x: f"{x:10.2f}"))
    med = res.groupby("kind")["F"].median()
    print(f"\nmedian F -- {dict(med.round(2))}")
    print(f"top metric overall: {res.iloc[0]['metric']} ({res.iloc[0]['kind']})")
    print("\nIf spatial does not beat bulk, that is a real finding. Report it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sweep_config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run"); r.add_argument("n", type=int)
    sub.add_parser("aggregate")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    {"plan": cmd_plan, "run": cmd_run, "aggregate": cmd_aggregate}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
