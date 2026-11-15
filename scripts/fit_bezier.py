"""Fit a cubic Bézier curve on R^3 x SO(3) to Lift or Tool Hang demonstrations and write the controls and decoded actions."""

import os

# Must be set before JAX starts. Small fits run faster on the CPU than with GPU call overhead.
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import jax
import numpy as np
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from bezier_rfm.datasets.robomimic import demo_keys, require
from bezier_rfm.fitting.bezier import decode, knot_times, phase_h
from bezier_rfm.fitting.density import place_capped, release_weight, speed_weight
from bezier_rfm.fitting.fit import CONTROL_DT, TRF_OPTIONS, fit, initialize


def phase_ks(durations, density):
    """Return the segment count of each phase for a segment density.

    Args:
        durations (np.ndarray): Phase durations T_i in s, shape (n,).
        density (float): Segments per second.

    Returns:
        tuple: K_i = max(1, round(density * T_i)), length n.
    """
    return tuple(max(1, int(round(density * d))) for d in durations)


def fit_demo(actions, ks=None, density=None, variant="event", rule=None):
    """Fit one demonstration and decode its curve back to absolute actions.

    Args:
        actions (np.ndarray): q0-start absolute actions, shape (N, 7).
        ks (tuple | None): Segments per phase of the event knots; None uses density.
        density (float | None): Segments per second, used when ks is None.
        variant (str): "event" (one knot at every gripper event), "uniform" (equal segments, same total K),
            "release" or "speed" (event knots, same total K, segments placed by a weight).
        rule (dict | None): Weight parameters: m, last (release), v0 (speed), h_min.

    Returns:
        tuple: Controls, segment durations h (s), ks, event indices, decoded actions (N, 7), and a diagnostics dict.
    """
    n = len(actions)
    p, r = actions[:, :3], Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
    events = (np.flatnonzero(np.diff(actions[:, 6])) + 1).tolist()
    bounds = [0, *events, n - 1]
    durations = CONTROL_DT * np.diff(bounds)
    event_ks = tuple(ks) if ks is not None else phase_ks(durations, density)
    require(len(event_ks) == len(events) + 1, "ks must have one entry per phase")
    rule = rule or {}
    if variant == "uniform":
        knot_events, fit_ks = [], (sum(event_ks),)
        h = np.asarray(phase_h(np.array([durations.sum()]), fit_ks))
    elif variant == "event":
        knot_events, fit_ks = events, event_ks
        h = np.asarray(phase_h(durations, fit_ks))
    else:
        if variant == "release":
            w = release_weight(actions, events, rule.get("m", 3.0), rule.get("last", 2.0))
        else:
            w = speed_weight(actions, rule.get("v0", 0.02))
        knot_events = events
        h, fit_ks, cap = place_capped(w, events, sum(event_ks), rule.get("h_min", 0.15))
    start = time.perf_counter()
    c0 = initialize(p, r, knot_events, h, fit_ks)
    t = CONTROL_DT * np.arange(n)
    c, diag = fit(c0, h, t[1:-1], p[1:-1], r[1:-1], n)
    p_hat, r_hat = (np.asarray(a) for a in decode(c, h, t))
    diag["seconds"] = time.perf_counter() - start
    if variant in ("release", "speed"):
        diag["weight_cap"] = cap
    if variant == "uniform":
        # The gripper events are not knots, so the stored column is kept unchanged.
        gripper = actions[:, 6].copy()
    else:
        # The gripper switches at the event knots: each phase keeps the command at its first action.
        tau = np.asarray(knot_times(h))
        knot_actions = np.round(tau[np.concatenate([[0], np.cumsum(fit_ks)])] / CONTROL_DT).astype(int)
        gripper = actions[knot_actions[np.searchsorted(knot_actions, np.arange(n), side="right") - 1], 6]
        require(np.array_equal(gripper, actions[:, 6]), "the gripper rule does not reproduce the stored gripper column")
    decoded = np.concatenate([p_hat, Rotation.from_matrix(r_hat).as_rotvec(), gripper[:, None]], axis=1)
    pos_err = np.linalg.norm(p_hat - p, axis=1)
    rot_err = Rotation.from_matrix(np.swapaxes(r_hat, 1, 2) @ r).magnitude()
    diag.update({
        "pos_rms_m": float(np.sqrt(np.mean(pos_err**2))),
        "pos_max_m": float(pos_err.max()),
        "rot_rms_rad": float(np.sqrt(np.mean(rot_err**2))),
        "rot_max_rad": float(rot_err.max()),
        "end_error": float(max(pos_err[0], pos_err[-1], rot_err[0], rot_err[-1])),
    })
    return c, h, fit_ks, events, decoded, diag


def main():
    """Fit the demonstrations for one knot rule and write the output file.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["lift", "tool_hang"], default="lift")
    root = Path(f"data/robomimic/datasets/{parser.parse_known_args()[0].task}/ph")
    parser.add_argument("--actions", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--ks", type=int, nargs="+", default=None, help="Segments per phase, the same for every demo (Lift: 5 2)")
    parser.add_argument("--density", type=float, default=None, help="Segments per second, per demo and phase (used without --ks)")
    parser.add_argument("--variant", choices=["event", "uniform", "release", "speed"], default="event")
    parser.add_argument("--m", type=float, default=3.0, help="release: weight before a release")
    parser.add_argument("--last", type=float, default=2.0, help="release: weighted window before a release, s")
    parser.add_argument("--v0", type=float, default=0.02, help="speed: speed offset, m/s")
    parser.add_argument("--h-min", type=float, default=0.15, help="release, speed: shortest segment, s")
    parser.add_argument("--demos", nargs="+", default=["all"], help='Demonstration names, or "all"')
    parser.add_argument("--output", type=Path, default=None, help="Default: bezier_k{ks}.hdf5 or bezier_{variant}_rho{density}.hdf5 next to the actions file")
    args = parser.parse_args()
    require((args.ks is None) != (args.density is None), "give exactly one of --ks and --density")
    if args.ks is not None:
        name = f"bezier_k{'_'.join(map(str, args.ks))}" + ("_uniform" if args.variant == "uniform" else "")
    else:
        name = f"bezier_{args.variant}_rho{args.density:g}"
    rule = {"m": args.m, "last": args.last, "v0": args.v0, "h_min": args.h_min} if args.variant in ("release", "speed") else None
    output = args.output or args.actions.with_name(f"{name}.hdf5")
    require(not output.exists(), f"{output} exists. Delete it first to write it again")

    temporary = output.with_name(output.name + ".tmp")
    with h5py.File(args.actions, "r") as f, h5py.File(temporary, "w") as out:
        data = f["data"]
        names = demo_keys(data) if args.demos == ["all"] else args.demos
        out_data = out.create_group("data")
        out_data.attrs["env_args"] = data.attrs["env_args"]
        out_data.attrs["bezier"] = json.dumps({
            "ks": args.ks,
            "density": args.density,
            "variant": args.variant,
            "rule": rule,
            "source": args.actions.name,
            "source_sha256": hashlib.sha256(args.actions.read_bytes()).hexdigest(),
            "trf": TRF_OPTIONS,
            "fit_target": "commanded poses of the source; q_0 = action 0 (start pose) and q_K = last action are fixed",
        })
        start = time.perf_counter()
        for i, name in enumerate(names):
            c, h, ks, events, decoded, diag = fit_demo(data[f"{name}/actions"][:], args.ks, args.density, args.variant, rule)
            group = out_data.create_group(name)
            group.create_dataset("actions", data=decoded)
            bezier = group.create_group("bezier")
            for key, value in zip(("q_pos", "q_rot", "H_pos", "H_rot"), c):
                bezier.create_dataset(key, data=value)
            for key, value in (("h", h), ("ks", ks), ("events", events), ("variant", args.variant)):
                bezier.attrs[key] = value
            bezier.attrs["diagnostics"] = json.dumps(diag)
            if i % 25 == 0:
                print(f"Finished {name}: status {diag['status']}, charts {diag['charts']}, pos RMS {1000 * diag['pos_rms_m']:.1f} mm ({time.perf_counter() - start:.0f} s)", flush=True)
    # The rename means a failed run never leaves a partial file at the output path.
    os.replace(temporary, output)
    output.chmod(0o400)
    print(f"Wrote {len(names)} fits to {output} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
