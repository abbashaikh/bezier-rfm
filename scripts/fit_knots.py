"""Fit Tool Hang demos with the H26 knot rules (S: segment clock, E: error-driven insertion), or with S on shifted tick grids (H27)."""

import os

# Must be set before JAX starts. Small fits run faster on the CPU than with GPU call overhead.
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import h5py
import jax
import numpy as np
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from bezier_rfm.fitting.bezier import decode
from bezier_rfm.fitting.density import speed_weight
from bezier_rfm.fitting.fit import CONTROL_DT, fit, initialize
from bezier_rfm.fitting.knots import clock_h, error_knots, euclidean_proxy, grid_tau


def rule_h(actions, rule, c, h_min):
    """Return the segment durations of one knot rule; E gets the K of S.

    Args:
        actions (np.ndarray): q0-start absolute actions, shape (N, 7).
        rule (str): "S" or "E".
        c (float): Clock constant of S.
        h_min (float): Shortest segment, s.

    Returns:
        np.ndarray: Segment durations h (s), shape (K,).
    """
    h = clock_h(speed_weight(actions), c, h_min)
    return h if rule == "S" else error_knots(euclidean_proxy(actions), len(h), h_min)


def fit_job(job):
    """Fit one demo with one rule (or one S grid) and decode it on the covered actions.

    Args:
        job (tuple): Actions path, demo name, rule, c, h_min, grid offset in ticks (None for the H26 rules).

    Returns:
        tuple: Demo name, rule, h (K,), controls, decoded actions (M, 7), diagnostics dict.
    """
    path, name, rule, c, h_min, offset = job
    with h5py.File(path, "r") as f:
        actions = f[f"data/{name}/actions"][:]
    n = len(actions)
    p, r = actions[:, :3], Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
    if offset is None:
        h, t0 = rule_h(actions, rule, c, h_min), 0.0
    else:
        tau = grid_tau(speed_weight(actions), c, offset, h_min)
        h, t0 = np.diff(tau), tau[0]
    start = time.perf_counter()
    c0 = initialize(p, r, [], h, (len(h),), t0)
    t = CONTROL_DT * np.arange(n)
    # Fit targets are the commands strictly inside (t0, t_end), in time relative to knot 0.
    first = int(np.floor(t0 / CONTROL_DT + 1e-9)) + 1
    ctrl, diag = fit(c0, h, t[first:-1] - t0, p[first:-1], r[first:-1], n - first + 1)
    # Decoded on the covered commands, first .. N-1 (for grid offset 0, first = 1 and action 0 is q_0).
    covered = slice(first - 1 if t0 == 0 else first, n)
    p_hat, r_hat = (np.asarray(a) for a in decode(ctrl, h, t[covered] - t0))
    # The gripper events are not knots, so the stored gripper column is kept unchanged.
    decoded = np.concatenate([p_hat, Rotation.from_matrix(r_hat).as_rotvec(), actions[covered, 6:7]], axis=1)
    pos_err = np.linalg.norm(p_hat - p[covered], axis=1)
    rot_err = Rotation.from_matrix(np.swapaxes(r_hat, 1, 2) @ r[covered]).magnitude()
    diag.update({
        "seconds": time.perf_counter() - start,
        "pos_rms_m": float(np.sqrt(np.mean(pos_err**2))),
        "rot_rms_rad": float(np.sqrt(np.mean(rot_err**2))),
        "t0": float(t0),
        "first_action": covered.start,
    })
    return name, rule, h, ctrl, decoded, diag


def grid_job(job):
    """Fit one demo on one S grid and write it to its own file, so that an interrupted run can resume.

    Args:
        job (tuple): Output path, then the fit_job tuple.

    Returns:
        str: Summary line.
    """
    out, *fit_args = job
    name, _, h, ctrl, decoded, diag = fit_job(tuple(fit_args))
    temporary = out.with_name(out.name + ".tmp")
    with h5py.File(temporary, "w") as g:
        for key, value in zip(("q_pos", "q_rot", "H_pos", "H_rot"), ctrl):
            g.create_dataset(key, data=value)
        g.create_dataset("h", data=h)
        g.create_dataset("actions", data=decoded)
        g.attrs.update({"demo": name, "offset": fit_args[5], "c": fit_args[3], "h_min": fit_args[4],
                        "t0": diag["t0"], "first_action": diag["first_action"], "diagnostics": json.dumps(diag)})
    os.replace(temporary, out)
    return (f"{out.name}: K {len(h)}, t0 {diag['t0']:.3f} s, status {diag['status']}, pos RMS {1000 * diag['pos_rms_m']:.2f} mm, "
            f"rot RMS {1000 * diag['rot_rms_rad']:.2f} mrad ({diag['seconds']:.0f} s)")


def main():
    """Fit all jobs in parallel and write bezier_<tag>_<rule>.hdf5, or one file per demo and grid.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/tool_hang/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--demos", nargs="+", default=["demo_189", "demo_82", "demo_66", "demo_16", "demo_137"],
                        help='Demonstration names, or "all"')
    parser.add_argument("--c", type=float, default=3.10)
    parser.add_argument("--h-min", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rules", nargs="+", default=["S", "E"], choices=["S", "E"])
    parser.add_argument("--tag", default="h26", help="Output files are bezier_<tag>_<rule>.hdf5")
    parser.add_argument("--grids", type=int, default=0, help="S only: fit on this many shifted grids, offsets i / grids")
    parser.add_argument("--out-dir", type=Path, help="With --grids: one file <demo>_g<i>.hdf5 per fit; existing files are skipped")
    args = parser.parse_args()
    if args.demos == ["all"]:
        with h5py.File(args.actions, "r") as f:
            # Numeric order; bezier_rfm.datasets.robomimic.demo_keys would import robomimic in every worker.
            args.demos = sorted(f["data"].keys(), key=lambda key: int(key.split("_")[1]))
    if args.grids:
        if args.out_dir is None:
            parser.error("--grids needs --out-dir")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(args.out_dir / f"{name}_g{i}.hdf5", str(args.actions), name, "S", args.c, args.h_min, i / args.grids)
                for name in args.demos for i in range(args.grids)]
        jobs = [job for job in jobs if not job[0].exists()]
        print(f"{len(jobs)} fits to run", flush=True)
        with get_context("spawn").Pool(args.workers) as pool:
            for line in pool.imap_unordered(grid_job, jobs):
                print(line, flush=True)
        return
    outputs = {rule: root / f"bezier_{args.tag}_{rule}.hdf5" for rule in args.rules}
    for out in outputs.values():
        if out.exists():
            raise FileExistsError(f"{out} exists. Delete it first to write it again")

    jobs = [(str(args.actions), name, rule, args.c, args.h_min, None) for rule in args.rules for name in args.demos]
    # JAX is not fork-safe once started, so the workers are spawned.
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(fit_job, jobs)
    with h5py.File(args.actions, "r") as f:
        env_args = f["data"].attrs["env_args"]
    for rule, out in outputs.items():
        temporary = out.with_name(out.name + ".tmp")
        with h5py.File(temporary, "w") as g:
            data = g.create_group("data")
            data.attrs["env_args"] = env_args
            data.attrs["bezier"] = json.dumps({"rule": rule, "c": args.c, "h_min": args.h_min})
            for name, r, h, ctrl, decoded, diag in results:
                if r != rule:
                    continue
                group = data.create_group(name)
                group.create_dataset("actions", data=decoded)
                bezier = group.create_group("bezier")
                for key, value in zip(("q_pos", "q_rot", "H_pos", "H_rot"), ctrl):
                    bezier.create_dataset(key, data=value)
                bezier.attrs["h"] = h
                bezier.attrs["diagnostics"] = json.dumps(diag)
                print(f"{rule} {name}: K {len(h)}, status {diag['status']}, pos RMS {1000 * diag['pos_rms_m']:.2f} mm, "
                      f"rot RMS {1000 * diag['rot_rms_rad']:.2f} mrad ({diag['seconds']:.0f} s)", flush=True)
        os.replace(temporary, out)
        out.chmod(0o400)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
