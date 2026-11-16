"""Fit Tool Hang demos with the H26 knot rules (S: segment clock, E: error-driven insertion) and write one file per rule."""

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
from bezier_rfm.fitting.knots import clock_h, error_knots, euclidean_proxy


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
    """Fit one demo with one rule and decode it on the action grid.

    Args:
        job (tuple): Actions path, demo name, rule, c, h_min.

    Returns:
        tuple: Demo name, rule, h (K,), controls, decoded actions (N, 7), diagnostics dict.
    """
    path, name, rule, c, h_min = job
    with h5py.File(path, "r") as f:
        actions = f[f"data/{name}/actions"][:]
    n = len(actions)
    p, r = actions[:, :3], Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
    h = rule_h(actions, rule, c, h_min)
    start = time.perf_counter()
    c0 = initialize(p, r, [], h, (len(h),))
    t = CONTROL_DT * np.arange(n)
    ctrl, diag = fit(c0, h, t[1:-1], p[1:-1], r[1:-1], n)
    p_hat, r_hat = (np.asarray(a) for a in decode(ctrl, h, t))
    # The gripper events are not knots, so the stored gripper column is kept unchanged.
    decoded = np.concatenate([p_hat, Rotation.from_matrix(r_hat).as_rotvec(), actions[:, 6:7]], axis=1)
    pos_err = np.linalg.norm(p_hat - p, axis=1)
    rot_err = Rotation.from_matrix(np.swapaxes(r_hat, 1, 2) @ r).magnitude()
    diag.update({
        "seconds": time.perf_counter() - start,
        "pos_rms_m": float(np.sqrt(np.mean(pos_err**2))),
        "rot_rms_rad": float(np.sqrt(np.mean(rot_err**2))),
    })
    return name, rule, h, ctrl, decoded, diag


def main():
    """Fit all jobs in parallel and write bezier_h26_{S,E}.hdf5.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/tool_hang/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--demos", nargs="+", default=["demo_189", "demo_82", "demo_66", "demo_16", "demo_137"])
    parser.add_argument("--c", type=float, default=3.10)
    parser.add_argument("--h-min", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    outputs = {rule: root / f"bezier_h26_{rule}.hdf5" for rule in ("S", "E")}
    for out in outputs.values():
        if out.exists():
            raise FileExistsError(f"{out} exists. Delete it first to write it again")

    jobs = [(str(args.actions), name, rule, args.c, args.h_min) for rule in ("S", "E") for name in args.demos]
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
