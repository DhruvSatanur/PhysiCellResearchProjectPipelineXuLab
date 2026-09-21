#!/usr/bin/env python3
"""
cell_distance_metrics.py
========================
The core spatial metrics for the project, written so they run on ANY table of
cells with x/y(/z) + a type + an ID:
  - on an initial-conditions CSV (treated as one snapshot), and
  - unchanged, on real PhysiCell output loaded with pcdl.

Metrics:
  1. neighbor_distances(df, radius)   -> the "Query/Neighbour/Distance" table.
  2. local_niche(df, radius)          -> per-cell neighbor count, density, and
                                         fraction of each type nearby (the niche).
  3. displacement_by_type(df0, df1)   -> matches the SAME cell ID across frames.
  4. displacement_series(frames)      -> displacement from frame 0 for every frame.

CHANGES FROM THE ORIGINAL VERSION (all driven by turning proliferation,
death and cell transformation ON for the tumor-demo replication):

  (a) pcdl returns the cell ID as the DataFrame *index*, not a column. The old
      _normalize only looked in df.columns, silently fell through to
      np.arange(len(df)), and then "matched" cells BY ROW POSITION. With a
      fixed population that accidentally gave the right answer. With births and
      deaths it silently returns wrong displacements -- no error, plausible
      numbers. _normalize now resets the index first, and displacement refuses
      to run on synthetic IDs at all.
  (b) Cell type can change mid-run (macrophage -> M1 macrophage). The old code
      labelled each cell with its type in the FIRST frame only. Displacement
      now reports type_start, type_end and a transformed flag.
  (c) Dead cells persist in output for a while after death and were being
      counted as neighbors, inflating local density. Added exclude_dead.

Also replaced the O(N^2) Python pair loops with cKDTree.query_pairs, which
matters once a growing tumor pushes frames past a few thousand cells.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def _normalize(df, require_real_ids=False, exclude_dead=False):
    df = df.reset_index()

    def pick(cands, default=None, required=True):
        for c in cands:
            if c in df.columns:
                return c
        if required and default is None:
            raise KeyError(f"None of {cands} found in columns {list(df.columns)}")
        return default

    xcol = pick(["x", "position_x", "pos_x"])
    ycol = pick(["y", "position_y", "pos_y"])
    zcol = pick(["z", "position_z", "pos_z"], default=None, required=False)
    tcol = pick(["cell_type", "type", "cell_type_name"])
    icol = pick(["ID", "id", "cell_ID"], default=None, required=False)
    dcol = pick(["dead"], default=None, required=False)

    if icol is None and require_real_ids:
        raise ValueError(
            "no ID column found, so cells cannot be matched across frames.\n"
            "Positional matching is only valid when the population is fixed; with\n"
            "births or deaths it silently returns wrong displacements.\n"
            f"columns present: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "x": df[xcol].astype(float),
        "y": df[ycol].astype(float),
        "z": df[zcol].astype(float) if zcol else 0.0,
        "type": df[tcol].astype(str),
    })
    out["ID"] = df[icol].values if icol is not None else np.arange(len(df))
    out["dead"] = df[dcol].values.astype(bool) if dcol is not None else False

    if exclude_dead:
        out = out[~out["dead"]]

    return out.reset_index(drop=True)


def _coords(d, dims):
    return d[["x", "y"]].values if dims == 2 else d[["x", "y", "z"]].values



def neighbor_distances(df, radius, dims=2, exclude_dead=False):
    """For each cell, every other cell within `radius`, with euclidean distance.

    Returns: query_id, query_type, neighbor_id, neighbor_type, distance.
    Each unordered pair appears twice (once per cell as the query), matching the
    table format in the project doc.
    """
    d = _normalize(df, exclude_dead=exclude_dead)
    if len(d) < 2:
        return pd.DataFrame(columns=["query_id", "query_type", "neighbor_id",
                                     "neighbor_type", "distance"])
    coords = _coords(d, dims)
    tree = cKDTree(coords)

    pairs = tree.query_pairs(r=radius, output_type="ndarray")   # unique i<j
    if len(pairs) == 0:
        return pd.DataFrame(columns=["query_id", "query_type", "neighbor_id",
                                     "neighbor_type", "distance"])
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(coords[i] - coords[j], axis=1)

    ids, types = d["ID"].values, d["type"].values
    q = np.concatenate([i, j])          # symmetrize so every cell is a query
    n = np.concatenate([j, i])
    return pd.DataFrame({
        "query_id": ids[q], "query_type": types[q],
        "neighbor_id": ids[n], "neighbor_type": types[n],
        "distance": np.concatenate([dist, dist]),
    })



def local_niche(df, radius, dims=2, exclude_dead=False):
    """Per-cell: neighbor count, local density, and fraction of each nearby type.

    'Density' = neighbors / area (2D, pi*r^2) or volume (3D).
    The frac_* columns ARE the local niche each cell experiences.

    Note the density is not edge-corrected: cells near the domain boundary have
    part of their neighborhood outside the domain and read low. That is fine for
    comparing cells within a run, but do not compare raw densities across
    domains of different size.
    """
    d = _normalize(df, exclude_dead=exclude_dead)
    coords = _coords(d, dims)
    tree = cKDTree(coords)
    types = sorted(d["type"].unique())
    measure = np.pi * radius**2 if dims == 2 else (4.0/3.0) * np.pi * radius**3

    counts = np.zeros(len(d), dtype=int)
    frac = {t: np.zeros(len(d)) for t in types}
    tvals = d["type"].values

    if len(d) >= 2:
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            np.add.at(counts, i, 1)
            np.add.at(counts, j, 1)
            for t in types:
                hits = np.zeros(len(d))
                np.add.at(hits, i, (tvals[j] == t).astype(float))
                np.add.at(hits, j, (tvals[i] == t).astype(float))
                frac[t] = hits

    out = d.copy()
    out["n_neighbors"] = counts
    out["local_density"] = counts / measure
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in types:
            out[f"frac_{t}"] = np.where(counts > 0, frac[t] / np.maximum(counts, 1), 0.0)
    return out



def displacement_by_type(df_start, df_end, dims=2):
    """Match the SAME cell ID in two frames and measure how far it moved.

    Only cells present in BOTH frames are reported, so births and deaths are
    excluded rather than producing spurious displacements.

    Because a cell's type can change mid-run (the tumor-demo model transforms
    macrophage -> M1 macrophage), this reports type_start and type_end. Group by
    type_start to ask 'how far did the cells that began as X travel', by
    type_end to ask 'what are the cells that are now Y'. They are not the same
    question, and using the wrong one silently mislabels every transformed cell.
    """
    a = _normalize(df_start, require_real_ids=True).set_index("ID")
    b = _normalize(df_end,   require_real_ids=True).set_index("ID")

    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("duplicate cell IDs within a frame -- frames may have "
                         "been concatenated, or output/ contains a stale run")

    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("no shared cell IDs between the two frames")

    cols = ["x", "y"] if dims == 2 else ["x", "y", "z"]
    delta = b.loc[common, cols].values - a.loc[common, cols].values
    t0 = a.loc[common, "type"].values
    t1 = b.loc[common, "type"].values

    out = pd.DataFrame({
        "ID": common,
        "type_start": t0,
        "type_end": t1,
        "transformed": t0 != t1,
        "displacement": np.linalg.norm(delta, axis=1),
        "dx": delta[:, 0],
        "dy": delta[:, 1],
        "died": b.loc[common, "dead"].values & ~a.loc[common, "dead"].values,
    })
    out["type"] = out["type_start"]      # back-compat with the old column name
    return out


def survival_summary(df_start, df_end):
    """How much of the population persisted. Report this alongside displacement:
    a mean over 40% of the starting cells means something different from a mean
    over 99%, and the number alone does not say which you have."""
    a = _normalize(df_start, require_real_ids=True)
    b = _normalize(df_end,   require_real_ids=True)
    sa, sb = set(a["ID"]), set(b["ID"])
    return {
        "n_start": len(sa), "n_end": len(sb),
        "n_tracked": len(sa & sb),
        "n_lost": len(sa - sb), "n_born": len(sb - sa),
        "frac_tracked": len(sa & sb) / max(len(sa), 1),
    }


def displacement_series(frames, dims=2):
    """Displacement from frame 0 for each later frame, aggregated by start type.

    `frames` is a list of (time, dataframe). Use load_pcdl_series() to build it.
    """
    rows = []
    t0, f0 = frames[0]
    for t, f in frames[1:]:
        d = displacement_by_type(f0, f, dims=dims)
        g = d.groupby("type_start")["displacement"]
        for typ, mean in g.mean().items():
            rows.append({"time": t, "type_start": typ,
                         "mean_displacement": mean,
                         "n_tracked": int(g.count()[typ])})
    return pd.DataFrame(rows)


def load_pcdl_series(output_dir):
    """Load a PhysiCell output folder as [(time, dataframe), ...].

    Keeps the ID as a real column, which _normalize depends on.
    """
    import pcdl
    ts = pcdl.TimeSeries(output_dir, verbose=False)
    return [(m.get_time(), m.get_cell_df().reset_index()) for m in ts.get_mcds_list()]



def _demo():
    R = 30.0
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "ID": np.arange(n),
        "x": rng.uniform(-400, 400, n),
        "y": rng.uniform(-400, 400, n),
        "z": 0.0,
        "type": rng.choice(["cancer", "immune"], n),
    })

    nd = neighbor_distances(df, radius=R)
    print(f"[1] neighbor pairs within {R:.0f} um: {len(nd)} rows")
    assert (nd.distance <= R + 1e-9).all()
    assert len(nd) % 2 == 0, "table should be symmetric"

    niche = local_niche(df, radius=R)
    fcols = [c for c in niche.columns if c.startswith("frac_")]
    print(f"[2] niche for {len(niche)} cells, mean neighbors "
          f"{niche.n_neighbors.mean():.2f}, type columns {fcols}")
    nz = niche.n_neighbors > 0
    assert np.allclose(niche.loc[nz, fcols].sum(axis=1), 1.0), "fractions must sum to 1"
    
    assert niche.n_neighbors.sum() == len(nd)

    # displacement
    f0 = pd.DataFrame({"ID": [1, 2, 3, 4], "x": [0, 0, 0, 0], "y": [0, 0, 0, 0],
                       "z": 0.0, "type": ["cancer", "macrophage", "macrophage", "cancer"]})
    f1 = pd.DataFrame({"ID": [1, 2, 3, 5], "x": [3, 0, 5, 9], "y": [4, 0, 12, 9],
                       "z": 0.0, "type": ["cancer", "macrophage", "M1 macrophage", "cancer"]})
    disp = displacement_by_type(f0, f1)
    assert sorted(disp.displacement) == [0.0, 5.0, 13.0]        # 3-4-5 and 5-12-13
    assert len(disp) == 3, "cell 4 died and cell 5 was born; both excluded"
    assert disp.set_index("ID").loc[3, "transformed"]
    assert not disp.set_index("ID").loc[2, "transformed"]
    print(f"[3] displacement self-test PASSED: {len(disp)} tracked, "
          f"1 transformed, birth and death correctly excluded")
    print(f"    survival: {survival_summary(f0, f1)}")


    no_id = f0.drop(columns=["ID"])
    try:
        displacement_by_type(no_id, f1.drop(columns=["ID"]))
        raise AssertionError("should have refused to match without real IDs")
    except ValueError as e:
        print(f"[4] refused positional matching, as intended")

    print("\nall self-tests passed")


if __name__ == "__main__":
    _demo()
