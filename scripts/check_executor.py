"""H29: run the grid-0 dataset windows through the receding executor as an oracle policy and check the commands."""

import os

# Small decodes run faster on the CPU than with GPU call overhead.
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from pathlib import Path

import h5py
import jax
import numpy as np
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from bezier_rfm.datasets.windows import CONTROL_DT
from bezier_rfm.fitting import bezier, so3
from bezier_rfm.policy.executor import Executor, grid_steps, to_commands

ROOT = Path("data/robomimic/datasets/tool_hang/ph")
WINDOW_KEYS = ("q_pos", "q_rot", "H_pos", "H_rot", "z", "logits", "v0", "gripper", "obs_dt", "t_start", "knot")


def load_demo(f, demo, grid=0):
    """Read the windows of one fit, ordered by start knot, and the extended fit.

    Args:
        f (h5py.File): Window file.
        demo (int): Demo number.
        grid (int): Clock grid.

    Returns:
        tuple: Windows dict (keys WINDOW_KEYS, leading axis W) and the extended fit dict.
    """
    w = f["windows"]
    sel = np.flatnonzero((w["demo"][:] == demo) & (w["grid"][:] == grid))
    sel = sel[np.argsort(w["knot"][sel])]
    windows = {key: w[key][sel] for key in WINDOW_KEYS}
    g = f[f"fits/demo_{demo}_g{grid}"]
    fit = {key: g[key][:] for key in ("q_pos", "q_rot", "H_pos", "H_rot", "h")}
    fit.update({key: g.attrs[key] for key in ("t0", "n_real")})
    return windows, fit


def oracle_commands(windows, q0_pos, q0_rot):
    """Execute the oracle chain: one window per knot, the whole last window, from the fit's start velocity.

    Args:
        windows (dict): Windows of one fit, ordered by start knot 0, 1, ...
        q0_pos (np.ndarray): Start position, shape (3,).
        q0_rot (np.ndarray): Start rotation, shape (3, 3).

    Returns:
        tuple: Actions (M, 7), the executor, and the obs_dt of each window call (W,).
    """
    n = len(windows["z"])
    if not np.array_equal(windows["knot"], np.arange(n)):
        raise ValueError("the windows do not start at every knot 0..W-1")
    ex = Executor(q0_pos, q0_rot, v0=windows["v0"][0])
    idx, cmds, obs_dt = [], [], []
    for j in range(n):
        obs_dt.append(ex.obs_frames()[1])
        i, c = ex.step({key: windows[key][j] for key in WINDOW_KEYS}, last=j == n - 1)
        idx.append(i)
        cmds.append(c)
    idx = np.concatenate(idx)
    if not np.array_equal(idx, np.arange(len(idx))):
        raise ValueError("the executor commands have gaps or repeats")
    return np.concatenate(cmds), ex, np.array(obs_dt)


def direct_commands(fit, t_end, raw_gripper, samples_per_seg=4):
    """Decode the extended fit at 20 Hz on [0, t_end], with the gripper sampled as in the windows.

    Args:
        fit (dict): Extended fit with q_pos, q_rot, H_pos, H_rot, h.
        t_end (float): Last time in s.
        raw_gripper (np.ndarray): Recorded gripper commands, shape (N,).
        samples_per_seg (int): Gripper samples per segment.

    Returns:
        np.ndarray: Actions, shape (M, 7).
    """
    c = bezier.Controls(*(fit[key] for key in ("q_pos", "q_rot", "H_pos", "H_rot")))
    h = fit["h"]
    t = CONTROL_DT * grid_steps(0.0, t_end, closed=True)
    pos, rot = (np.asarray(a) for a in bezier.decode(c, h, t))
    # Gripper samples of each fit segment: the latest recorded command at u = i / S, past the end the last command.
    tau = np.r_[0.0, np.cumsum(h)]
    t_s = (tau[:-1, None] + np.arange(samples_per_seg) / samples_per_seg * h[:, None]).reshape(-1)
    g_s = raw_gripper[np.minimum(np.floor(t_s / CONTROL_DT + 1e-9).astype(int), len(raw_gripper) - 1)]
    grip = g_s[np.searchsorted(t_s, t + 1e-9, side="right") - 1]
    return to_commands(pos, rot, np.where(grip > 0, 1.0, -1.0))


def segment_velocity(c, h, k, u):
    """Return the velocity in seconds of segment k at local parameter u.

    Args:
        c (bezier.Controls): Controls.
        h (np.ndarray): Durations, shape (K,).
        k (int): Segment index.
        u (float): Local parameter in [0, 1].

    Returns:
        np.ndarray: Linear and body-frame angular velocity, shape (6,).
    """
    pos, rot = bezier.segment_controls(c, jax.numpy.asarray(h))

    def point(s):
        return bezier.segment_point(pos[k], rot[k], s)

    (_, r), (dp, dr) = point(u), jax.jacfwd(point)(float(u))
    return np.r_[np.asarray(dp), np.asarray(so3.vee(r.T @ dr))] / h[k]


def handover_jumps(ex):
    """Return the velocity jump at every replan knot: end of segment 0 of call j against the start of call j + 1.

    Args:
        ex (Executor): Executor after the oracle chain.

    Returns:
        np.ndarray: Linear (m/s) and angular (rad/s) jump norms, shape (W - 1, 2).
    """
    out = []
    for a, b in zip(ex.log[:-1], ex.log[1:]):
        jump = segment_velocity(a["controls"], a["h"], 0, 1.0) - segment_velocity(b["controls"], b["h"], 0, 0.0)
        out.append([np.linalg.norm(jump[:3]), np.linalg.norm(jump[3:])])
    return np.array(out)


def command_errors(a, b):
    """Return per-step position and rotation differences and the number of unequal gripper commands.

    Args:
        a (np.ndarray): Actions, shape (M, 7).
        b (np.ndarray): Actions, shape (M, 7).

    Returns:
        tuple: Position differences (M,) in m, rotation differences (M,) in rad, unequal gripper count.
    """
    dp = np.linalg.norm(a[:, :3] - b[:, :3], axis=1)
    dr = (Rotation.from_rotvec(a[:, 3:6]).inv() * Rotation.from_rotvec(b[:, 3:6])).magnitude()
    return dp, dr, int(np.sum(a[:, 6] != b[:, 6]))


def offline_check(f, demos_file, demo):
    """Run the oracle chain of one demo offline and compare it with the direct decode.

    Args:
        f (h5py.File): Window file.
        demos_file (h5py.File): q0-start action file.
        demo (int): Demo number.

    Returns:
        dict: Check values and the two command streams.
    """
    windows, fit = load_demo(f, demo)
    actions = demos_file[f"data/demo_{demo}/actions"][:]
    # The actual start pose; the grid-0 fit fixes q_0 to it (action 0 of the q0-start file).
    q0_pos, q0_rot = actions[0, :3], Rotation.from_rotvec(actions[0, 3:6]).as_matrix()
    oracle, ex, obs_dt = oracle_commands(windows, q0_pos, q0_rot)
    direct = direct_commands(fit, ex.t, actions[:, 6])
    dp, dr, dg = command_errors(oracle, direct)
    jumps = handover_jumps(ex)
    n_raw = len(actions)
    return {
        "demo": demo, "windows": len(windows["z"]), "steps": len(oracle), "raw_steps": n_raw,
        "t_end": ex.t, "fit_T": float(fit["h"].sum()),
        "pos_err": float(dp.max()), "rot_err": float(dr.max()), "grip_diff": dg,
        "jump_lin": float(jumps[:, 0].max()), "jump_ang": float(jumps[:, 1].max()),
        "obs_dt_err": float(np.abs(obs_dt - windows["obs_dt"]).max()),
        "t_start_err": float(np.abs(np.array([e["t"] for e in ex.log]) - windows["t_start"]).max()),
        "grip_vs_raw": int(np.sum(direct[:n_raw, 6] != actions[:, 6])),
        "oracle": oracle, "direct": direct,
    }


def screening_noise(n, seed, pos_sigma, rot_sigma):
    """Return the command noise of scripts/screen_replay.py perturb for n actions, with row 0 zero.

    Args:
        n (int): Number of actions.
        seed (int): Random seed; 0 gives no noise.
        pos_sigma (float): Position noise per axis (m).
        rot_sigma (float): Rotation noise per axis (rad).

    Returns:
        tuple: Position noise (n, 3) and left-multiplied rotation-vector noise (n, 3).
    """
    pos, rot = np.zeros((n, 3)), np.zeros((n, 3))
    if seed:
        # Same draw order as perturb: positions first, then rotations.
        rng = np.random.default_rng(seed)
        pos[1:] = pos_sigma * rng.normal(size=(n - 1, 3))
        rot[1:] = rot_sigma * rng.normal(size=(n - 1, 3))
    return pos, rot


def write_direct(rows, demos_file, out):
    """Write the direct decodes as a screen_replay action file.

    Args:
        rows (list): Offline check results with the direct commands.
        demos_file (h5py.File): q0-start action file, for the environment metadata.
        out (Path): Output path.

    Returns:
        None: Nothing.
    """
    with h5py.File(out, "w") as g:
        data = g.create_group("data")
        data.attrs["env_args"] = demos_file["data"].attrs["env_args"]
        for r in rows:
            data.create_dataset(f"demo_{r['demo']}/actions", data=r["direct"])


def replay(args, f, rows):
    """Run the oracle through the rollout with the screening protocol and compare with the direct replays.

    Args:
        args (Namespace): Command-line arguments.
        f (h5py.File): Window file.
        rows (list): Offline check results.

    Returns:
        None: Nothing.
    """
    import json

    from bezier_rfm.eval.rollout import rollout

    windows = {r["demo"]: load_demo(f, r["demo"])[0] for r in rows}

    def oracle(requests):
        out = []
        for q in requests:
            w = windows[int(q["demo"].split("_")[1])]
            j = q["call"]
            out.append({**{key: w[key][j] for key in WINDOW_KEYS}, "last": j == len(w["z"]) - 1})
        return out

    jobs, expected = [], []
    for r in rows:
        for seed in range(args.seeds + 1):
            noise = screening_noise(len(r["oracle"]), seed, args.pos_sigma, args.rot_sigma)
            # Seed 0 sends the commands unchanged, as perturb does; a zero noise would still round the rotations.
            jobs.append({"demo": f"demo_{r['demo']}", "seed": seed, "v0": windows[r["demo"]]["v0"][0],
                         "noise": noise if seed else None})
            # The commands that screen_replay sends for the direct file: the direct decode with the same noise.
            a = r["direct"].copy()
            if seed:
                a[:, :3] += noise[0]
                a[:, 3:6] = (Rotation.from_rotvec(noise[1]) * Rotation.from_rotvec(a[:, 3:6])).as_rotvec()
            expected.append(a)
    with h5py.File(args.demos_file, "r") as d:
        env_args = d["data"].attrs["env_args"]
    results = rollout(env_args, str(args.source), jobs, oracle, n_envs=args.workers, hold=args.hold)

    direct = {(r["demo"], r["seed"]): r for r in json.loads(args.direct_json.read_text())["runs"]}
    h27 = {(r["demo"], r["seed"]): r for r in json.loads(args.h27_json.read_text())["runs"]}
    lines = ["| demo | seed | cmd pos err (m) | cmd rot err (rad) | grip diff | oracle | direct ext | H27 file |", "|---|---|---|---|---|---|---|---|"]
    runs = []
    for job, res, exp in zip(jobs, results, expected):
        dp, dr, dg = command_errors(res["actions"], exp)
        key = (job["demo"], job["seed"])
        runs.append({"demo": job["demo"], "seed": job["seed"], "oracle": res["success"], "first": res["first_success_step"],
                     "steps": res["steps"], "direct_ext": direct[key]["success_with_hold"], "h27": h27[key]["success_with_hold"],
                     "pos_err": float(dp.max()), "rot_err": float(dr.max()), "grip_diff": dg})
        r = runs[-1]
        lines.append(f"| {r['demo']} | {r['seed']} | {r['pos_err']:.1e} | {r['rot_err']:.1e} | {dg} | {int(r['oracle'])} | "
                     f"{int(r['direct_ext'])} | {int(r['h27'])} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"settings": {k: str(v) for k, v in vars(args).items()}, "runs": runs}, indent=1))
    print("\n".join(lines))
    tot = {k: sum(r[k] for r in runs) for k in ("oracle", "direct_ext", "h27")}
    same = sum(r["oracle"] == r["direct_ext"] for r in runs)
    print(f"Totals: oracle {tot['oracle']}, direct ext {tot['direct_ext']}, H27 file {tot['h27']} of {len(runs)}; "
          f"oracle = direct ext in {same} of {len(runs)} runs")


def main():
    """Run the offline command and handover checks, then optionally write the direct file or run the replay.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=ROOT / "windows_K3_c2.90_g4_all.hdf5")
    parser.add_argument("--demos-file", type=Path, default=ROOT / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--source", type=Path, default=ROOT / "low_dim_v15.hdf5")
    parser.add_argument("--demos", nargs="+", type=int, default=[189, 82, 66, 16, 137])
    parser.add_argument("--write-direct", type=Path, help="Write the direct extended-fit decodes for screen_replay.py")
    parser.add_argument("--replay", action="store_true", help="Run the oracle rollout and compare with the direct replays")
    parser.add_argument("--direct-json", type=Path, default=Path("docs/tool_hang/h29/direct_ext.json"))
    parser.add_argument("--h27-json", type=Path, default=Path("docs/tool_hang/screening/h27_S_c2.90.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/tool_hang/h29/oracle.json"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--pos-sigma", type=float, default=2e-4)
    parser.add_argument("--rot-sigma", type=float, default=2e-4)
    parser.add_argument("--hold", type=int, default=60)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.windows, "r") as f, h5py.File(args.demos_file, "r") as d:
        rows = [offline_check(f, d, demo) for demo in args.demos]
        cols = ("demo", "windows", "steps", "raw_steps", "pos_err", "rot_err", "grip_diff", "jump_lin", "jump_ang",
                "obs_dt_err", "t_start_err", "grip_vs_raw")
        print(" ".join(f"{c:>11}" for c in cols))
        for r in rows:
            print(" ".join(f"{r[c]:>11.2e}" if isinstance(r[c], float) else f"{r[c]:>11}" for c in cols), flush=True)
        if args.write_direct:
            write_direct(rows, d, args.write_direct)
            print(f"Wrote {args.write_direct}")
        if args.replay:
            replay(args, f, rows)


if __name__ == "__main__":
    main()
