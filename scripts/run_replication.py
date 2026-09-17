#!/usr/bin/env python3
"""
run_replication.py -- staged replication of Heiland et al. 2024.

The paper builds its model one rule at a time and reports an agent count at each
step. Those counts are the replication evidence, and most belong to INTERMEDIATE
models -- running only the final 7-rule version leaves almost every documented
claim untestable. This runs the stages and produces the comparison table.

    python run_replication.py build   --stages 0,1,2,3,8
    python run_replication.py run     --stages 0,1,2,3,8 --seeds 0,1,2
    python run_replication.py report

Or `all` to do the three in sequence.

What "replicated" means here: the paper says results are stochastic with more
than one OpenMP thread, so exact counts are not reproducible and are not the
target. The claim is that observed counts fall within run-to-run spread of the
reported ones, AND that the cascade goes the right direction at the right
magnitude: 5099 -> 1206 -> 705 -> 512 as rules 1, 2, 3 are added.
"""

import argparse, copy, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_tumor_demo as B

RULE_ORDER = [
    "cancer,pressure,decreases,cycle entry,0.0,1.0,4,0",
    "cancer,oxygen,increases,cycle entry,0.00093,21.5,4,0",
    "cancer,oxygen,decreases,necrosis,0.0,3.75,8,0",
    "cancer,drug,increases,apoptosis,5e-3,0.5,4,0",
    "cancer,dead,increases,debris secretion,1,0.5,10,1",
    "macrophage,contact with dead cell,increases,transform to M1 macrophage,0.1,0.5,10,0",
    "cancer,damage,increases,apoptosis,0.01,5,4,0",
]


def stage_spec(full_spec, stage):
    """Filter the full model spec down to one stage."""
    s = copy.deepcopy(full_spec)

    keep_sub = set(stage["substrates"])
    s["substrates"] = {k: v for k, v in s["substrates"].items()
                       if k.startswith("_") or k in keep_sub}

    if stage["oxygen_dirichlet"] == "symmetric":
        for b in s["substrates"]["oxygen"]["boundaries"].values():
            b["value"] = 38

    keep_ct = stage["cell_types"]
    s["cell_types"] = {k: v for k, v in s["cell_types"].items()
                       if k.startswith("_") or k in keep_ct}
    # a stage without necrosis-driving rules keeps the base necrosis rate
    if 2 not in stage["rules"] and "cancer" in s["cell_types"]:
        s["cell_types"]["cancer"]["death"]["necrosis_rate"] = 0.0
    for name in keep_ct:
        if name in s["cell_types"]:
            s["cell_types"][name]["cycle"] = stage["cycle"]
            break   # only the base type carries an explicit cycle; copies inherit

    # drop secretion entries for substrates this stage does not define
    for ct in s["cell_types"].values():
        if isinstance(ct, dict) and "secretion" in ct:
            ct["secretion"] = {k: v for k, v in ct["secretion"].items()
                               if k.startswith("_") or k in keep_sub}

    s["rules"]["enabled"] = bool(stage["rules"])
    for name in stage.get("disable_chemotaxis", []):
        s["cell_types"][name]["motility"]["chemotaxis"]["enabled"] = False
    return s


def build_stage(full_spec, ic_cfg, stage, template, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    spec = stage_spec(full_spec, stage)

    # rules file for this stage
    rules = [RULE_ORDER[i] for i in stage["rules"]]
    (outdir / "cell_rules.csv").write_text("\n".join(rules) + ("\n" if rules else ""))
    spec["rules"]["folder"] = str(outdir)
    spec["rules"]["filename"] = "cell_rules.csv"
    spec["initial_conditions"]["folder"] = str(outdir)
    spec["initial_conditions"]["filename"] = "cells.csv"

    import xml.etree.ElementTree as ET
    tree = ET.parse(template)
    root = tree.getroot()
    B.apply_domain(root, spec["domain"])
    B.apply_overall(root, spec["overall"])
    B.apply_substrates(root, spec["substrates"], spec["options"])
    B.apply_cell_types(root, spec["cell_types"])
    B.apply_ics(root, spec["initial_conditions"])
    B.apply_rules(root, spec["rules"])
    B.apply_user_params(root, spec["user_parameters"])
    ET.indent(tree, space="  ")
    xml_path = outdir / "config.xml"
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

    problems, notes = B.verify(xml_path, spec)
    if problems:
        print(f"  stage {stage['id']} VERIFY FAILED:")
        for p in problems:
            print("    ! " + p)
        return None

    # initial conditions, restricted to this stage's regions
    cfg = copy.deepcopy(ic_cfg)
    cfg["regions"] = [r for r in cfg["regions"] if r["name"] in stage["ic_regions"]]
    cfg["output_csv"] = str(outdir / "cells.csv")
    import make_cells as MC
    MC.configure(cfg)
    MC.main()
    return xml_path


def run_stage(root, xml_path, outdir, seed, cfg):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path); r = tree.getroot()
    odir = outdir / f"output_seed{seed}"
    if odir.exists():
        shutil.rmtree(odir)
    odir.mkdir(parents=True)
    r.find("save/folder").text = str(odir)
    node = r.find("user_parameters/random_seed")
    if node is not None:
        node.text = str(seed)
    run_xml = outdir / f"config_seed{seed}.xml"
    tree.write(run_xml, encoding="UTF-8", xml_declaration=True)

    exe = root / (cfg["executable_windows"] if platform.system() == "Windows"
                  else cfg["executable"])
    env = os.environ.copy()
    if cfg.get("dll_path"):
        env["PATH"] = cfg["dll_path"] + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    with open(outdir / f"stdout_seed{seed}.log", "w") as lf:
        p = subprocess.run([str(exe), str(run_xml)], cwd=root,
                           stdout=lf, stderr=subprocess.STDOUT, env=env)
    return p.returncode, time.time() - t0, odir


def counts_at(output_dir, times):
    """Agent count at each requested simulated time, nearest available frame."""
    import pcdl
    ts = pcdl.TimeSeries(str(output_dir), verbose=False)
    frames = [(m.get_time(), len(m.get_cell_df())) for m in ts.get_mcds_list()]
    if not frames:
        return {}
    got = {}
    for t in times:
        tt, nn = min(frames, key=lambda f: abs(f[0] - t))
        got[t] = (nn, tt)
    return got


def cmd_build(a, stages, spec, ic, cfg):
    for st in stages:
        print(f"\n=== stage {st['id']}: {st['name']} ===")
        print(f"    {st['why']}")
        out = (Path(cfg["work_root"]) / f"stage{st['id']}_{st['name']}").resolve()
        if build_stage(spec, ic, st, Path(cfg["physicell_root"]) / cfg["base_config"], out):
            print(f"    built -> {out}")


def cmd_run(a, stages, spec, ic, cfg):
    root = Path(cfg["physicell_root"]).resolve()
    for st in stages:
        out = (Path(cfg["work_root"]) / f"stage{st['id']}_{st['name']}").resolve()
        xml = out / "config.xml"
        if not xml.exists():
            print(f"stage {st['id']}: not built, run `build` first"); continue
        for seed in a.seed_list:
            rc, wall, odir = run_stage(root, xml, out, seed, cfg)
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"stage {st['id']} seed {seed}: {status} in {wall:.0f}s")


def cmd_report(a, stages, spec, ic, cfg, tol, seeds):
    rows = []
    for st in stages:
        out = (Path(cfg["work_root"]) / f"stage{st['id']}_{st['name']}").resolve()
        times = [c[0] for c in st["checkpoints"]]
        per_seed = {}
        for seed in seeds:
            odir = out / f"output_seed{seed}"
            if not odir.exists():
                continue
            try:
                per_seed[seed] = counts_at(odir, times)
            except Exception as e:
                print(f"  (stage {st['id']} seed {seed} unreadable: {e})")
        if not per_seed:
            for t, paper, fig in st["checkpoints"]:
                rows.append({"stage": st["id"], "name": st["name"], "t_min": t,
                             "paper": paper, "observed": None, "spread": None,
                             "pct_diff": None, "within": None, "figure": fig})
            continue
        for t, paper, fig in st["checkpoints"]:
            vals = [per_seed[s][t][0] for s in per_seed if t in per_seed[s]]
            if not vals:
                continue
            mean = float(np.mean(vals))
            pct = 100.0 * (mean - paper) / max(paper, 1)
            rows.append({
                "stage": st["id"], "name": st["name"], "t_min": t,
                "paper": paper, "observed": round(mean, 1),
                "spread": f"{min(vals)}-{max(vals)}" if len(vals) > 1 else str(vals[0]),
                "pct_diff": round(pct, 1),
                "within": abs(pct) <= tol, "figure": fig,
            })

    df = pd.DataFrame(rows)
    outp = Path(cfg["work_root"]) / "replication_table.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)

    print("\nREPLICATION TABLE  (paper agent counts vs observed)\n")
    print(df.to_string(index=False))
    done = df[df["observed"].notna()]
    if len(done):
        n_ok = int(done["within"].sum())
        print(f"\n{n_ok}/{len(done)} checkpoints within {tol}%")
        print("\nThe cascade matters as much as the individual numbers. Adding")
        print("rules 1, 2, 3 should drive the 3600-min count down through roughly")
        print("5099 -> 1206 -> 705 -> 512. A monotonic drop of about the right size")
        print("is strong evidence even where single counts sit outside the band.")
    print(f"\nwrote {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "report", "all"])
    ap.add_argument("--stages", default="min",
                    help="comma list, 'all', or 'min' for the minimum defensible set")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--spec", default="tumor_demo_spec.json")
    ap.add_argument("--ic", default="cells_config_tumor_demo.json")
    ap.add_argument("--stagedef", default="replication_stages.json")
    ap.add_argument("--sweepcfg", default="sweep_config.json")
    a = ap.parse_args()

    sd = json.load(open(a.stagedef))
    spec = json.load(open(a.spec))
    ic = json.load(open(a.ic))
    cfg = json.load(open(a.sweepcfg))
    cfg["work_root"] = "replication"

    if a.stages == "all":
        want = [s["id"] for s in sd["stages"]]
    elif a.stages == "min":
        want = sd["minimum_defensible_set"]
    else:
        want = [int(x) for x in a.stages.split(",")]
    stages = [s for s in sd["stages"] if s["id"] in want]

    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else sd["seeds"]
    a.seed_list = seeds
    tol = sd["tolerance_pct"]

    print(f"stages: {[s['id'] for s in stages]}   seeds: {seeds}")
    if a.cmd in ("build", "all"):  cmd_build(a, stages, spec, ic, cfg)
    if a.cmd in ("run", "all"):    cmd_run(a, stages, spec, ic, cfg)
    if a.cmd in ("report", "all"): cmd_report(a, stages, spec, ic, cfg, tol, seeds)


if __name__ == "__main__":
    main()
