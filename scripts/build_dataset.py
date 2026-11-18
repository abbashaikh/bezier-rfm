"""Build the H27 training windows from the per-grid clock fits of fit_knots.py --grids, and write one hdf5 file."""

import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np

from bezier_rfm.datasets.windows import OBS_KEYS, cut_windows, extend_fit, tail_travel, window_targets

WINDOW_KEYS = ("q_pos", "q_rot", "H_pos", "H_rot", "h", "z", "logits", "v0", "gripper", "obs", "obs_dt", "t_start",
               "demo", "grid", "knot", "n_tail")


def main():
    """Add braking tails, cut the windows of every fit file and write them with the extended fits.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/tool_hang/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-dir", type=Path, default=root / "fits_c2.90_g4")
    parser.add_argument("--demos-file", type=Path, default=root / "low_dim_abs_q0start.hdf5", help="Raw gripper commands and observations")
    parser.add_argument("--out", type=Path, default=root / "windows_K3_c2.90_g4.hdf5")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--a-max", type=float, default=0.8, help="Linear braking cap, m/s^2")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="Angular braking cap, rad/s^2")
    parser.add_argument("--max-pos", type=float, default=0.05, help="Largest tail travel, m")
    parser.add_argument("--max-rot-deg", type=float, default=5.0, help="Largest tail rotation, deg")
    parser.add_argument("--samples-per-seg", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists. Delete it first to write it again")

    files = sorted(args.fits_dir.glob("demo_*_g*.hdf5"), key=lambda p: tuple(map(int, re.findall(r"\d+", p.stem))))
    tail_args = {"a_max": args.a_max, "alpha_max": args.alpha_max, "max_pos": args.max_pos, "max_rot": np.radians(args.max_rot_deg)}
    parts, fits = [], {}
    with h5py.File(args.demos_file, "r") as demos:
        for path in files:
            with h5py.File(path, "r") as f:
                fit = {key: f[key][:] for key in ("q_pos", "q_rot", "H_pos", "H_rot", "h")}
                attrs = dict(f.attrs)
            name = attrs["demo"]
            grid = int(path.stem.split("_g")[-1])
            ext = extend_fit(fit, args.k - 1, h_min=attrs["h_min"], **tail_args)
            w = cut_windows(ext, args.k)
            pos, rot = tail_travel(ext, w)
            keep = (pos <= args.max_pos) & (rot <= tail_args["max_rot"])
            demo = demos[f"data/{name}"]
            obs = np.concatenate([demo[f"obs/{key}"][:] for key in OBS_KEYS], axis=1)
            w.update(window_targets(w, attrs["t0"], demo["actions"][:, 6], obs, args.samples_per_seg))
            w["demo"] = np.full(len(keep), int(name.split("_")[1]))
            w["grid"] = np.full(len(keep), grid)
            parts.append({key: w[key][keep] for key in WINDOW_KEYS})
            counts = {"full": int(np.sum(w["n_tail"] == 0)), "tail_kept": int(np.sum(keep & (w["n_tail"] > 0))),
                      "tail_dropped": int(np.sum(~keep))}
            fits[path.stem] = (ext, attrs, counts)
            print(f"{path.stem}: {counts}, t_b {ext['t_b']:.3f} s, full stop {ext['full_stop']}", flush=True)

    temporary = args.out.with_name(args.out.name + ".tmp")
    with h5py.File(temporary, "w") as g:
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        g.attrs["settings"] = json.dumps({**settings, "obs_keys": OBS_KEYS})
        windows = g.create_group("windows")
        for key in WINDOW_KEYS:
            windows.create_dataset(key, data=np.concatenate([p[key] for p in parts]))
        for stem, (ext, attrs, counts) in fits.items():
            group = g.create_group(f"fits/{stem}")
            for key in ("q_pos", "q_rot", "H_pos", "H_rot", "h"):
                group.create_dataset(key, data=ext[key])
            group.attrs.update({"demo": attrs["demo"], "offset": attrs["offset"], "c": attrs["c"], "h_min": attrs["h_min"],
                                "t0": attrs["t0"], "n_real": ext["n_real"], "t_b": ext["t_b"], "h_brake": ext["h_brake"],
                                "full_stop": ext["full_stop"], "counts": json.dumps(counts), "fit_diagnostics": attrs["diagnostics"]})
    os.replace(temporary, args.out)
    n = sum(len(p["z"]) for p in parts)
    print(f"Wrote {args.out}: {n} windows from {len(files)} fits")


if __name__ == "__main__":
    main()
