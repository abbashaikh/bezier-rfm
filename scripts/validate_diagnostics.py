"""Check diagnostic traces for time alignment, command fidelity, frames, completeness, rotation, and velocity."""

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import demo_keys, load_demo_model, make_env
from bezier_rfm.eval.lift_diagnostics import ACTION_FIELDS, CONTROL_DT, STATE_FIELDS, achieved_errors, command_fidelity, geodesic, read_trace, tracking_errors
from inspect_lift_demo import table

TIME_TOL = 1e-9
GOAL_TOL = 1e-8
FRAME_TOL = 1e-9
# obs/robot0_eef_quat_site is stored as float32, so it carries about 1e-7 rounding per component.
FRAME_ROT_TOL = 1e-6
ORTHO_TOL = 1e-9
# Stored site velocities must equal Jacobian x qvel at the stored state.
VELOCITY_TOL = 1e-10
REPRO_TOL = 1e-12
# A stored rotation-vector change above ROT_JUMP is a branch jump. Across it, the change of e_R must stay within
# the triangle-inequality bound: command rotation from t - 1 to t plus achieved rotation from t to t + 1.
ROT_JUMP = 1.0
ROT_SLACK_TOL = 1e-9
GEODESIC_TOL = 1e-6
ROTATION_FIELDS = ("ach_rot", "cube_rot", "cmd_rot", "goal_rot")


def angles(rot_a, rot_b):
    """Return geodesic angles between two stacks of rotation matrices.

    Args:
        rot_a (np.ndarray): Rotation matrices, shape (M, 3, 3).
        rot_b (np.ndarray): Rotation matrices, shape (M, 3, 3).

    Returns:
        np.ndarray: Angles in radians, shape (M,).
    """
    return Rotation.from_matrix(np.swapaxes(rot_a, 1, 2) @ rot_b).magnitude()


def check_time(trace, kind):
    """Return the largest timing errors of one trace.

    Args:
        trace (dict): Trace arrays.
        kind (str): "recorded" or "replay".

    Returns:
        dict: Interval-length error and state-time error in seconds.
    """
    n = len(trace["t_start"])
    times = trace["sim_time"][trace["valid"]]
    expected = times[0] + CONTROL_DT * np.arange(len(times))
    return {
        "interval": float(np.abs(trace["t_end"] - trace["t_start"] - CONTROL_DT).max()),
        "start_times": float(np.abs(trace["t_start"] - CONTROL_DT * np.arange(n)).max()),
        "state_times": float(np.abs(times - expected).max()),
        "valid_states": int(trace["valid"].sum()),
        "expected_valid": n if kind == "recorded" else n + 1,
    }


def check_completeness(trace):
    """Return shape, finiteness, NaN-padding, and orthonormality results of one trace.

    Args:
        trace (dict): Trace arrays.

    Returns:
        dict: Problems found (list) and the largest orthonormality error.
    """
    n = len(trace["t_start"])
    problems = []
    for fields, rows in ((STATE_FIELDS, n + 1), (ACTION_FIELDS, n)):
        for name, shape in fields.items():
            if trace[name].shape != (rows, *shape):
                problems.append(f"{name} has shape {trace[name].shape}")
    valid = trace["valid"]
    for name in STATE_FIELDS:
        if trace[name].dtype.kind == "f":
            if not np.isfinite(trace[name][valid]).all():
                problems.append(f"{name} has non-finite valid rows")
            if (~valid).any() and not np.isnan(trace[name][~valid]).all():
                problems.append(f"{name} has values in invalid rows")
    ortho = 0.0
    for name in ROTATION_FIELDS:
        rows = trace[name][valid] if name in STATE_FIELDS else trace[name]
        rows = rows[np.isfinite(rows).all(axis=(1, 2))]
        if len(rows):
            gram = np.swapaxes(rows, 1, 2) @ rows - np.eye(3)
            ortho = max(ortho, float(np.abs(gram).max()), float(np.abs(np.linalg.det(rows) - 1).max()))
    return {"problems": problems, "ortho": ortho}


def finite_difference_velocity(trace):
    """Compare sampled site velocities with finite differences of the poses, for information only.

    Args:
        trace (dict): Trace arrays.

    Returns:
        dict: Median relative errors of the linear and angular velocity.
    """
    k = np.flatnonzero(trace["valid"][:-1] & trace["valid"][1:])
    pos, rot = trace["ach_pos"], trace["ach_rot"]
    lin_fd = (pos[k + 1] - pos[k]) / CONTROL_DT
    # World-frame angular velocity from R_{k+1} R_k^T.
    ang_fd = Rotation.from_matrix(rot[k + 1] @ np.swapaxes(rot[k], 1, 2)).as_rotvec() / CONTROL_DT
    # Sampled velocities are instantaneous, so compare their mean over each interval.
    lin = 0.5 * (trace["ach_linvel"][k] + trace["ach_linvel"][k + 1])
    ang = 0.5 * (trace["ach_angvel"][k] + trace["ach_angvel"][k + 1])
    moving = np.linalg.norm(lin_fd, axis=1) > 0.02
    turning = np.linalg.norm(ang_fd, axis=1) > 0.05
    rel = lambda a, b, m: float(np.median(np.linalg.norm(a[m] - b[m], axis=1) / np.linalg.norm(b[m], axis=1))) if m.any() else float("nan")
    return {"linear": rel(lin, lin_fd, moving), "angular": rel(ang, ang_fd, turning), "moving_steps": int(moving.sum()), "turning_steps": int(turning.sum())}


def check_reproducibility(recorded, replay, start, name, experiment_b):
    """Compare the replay with the saved Experiment B results of the same start.

    Args:
        recorded (dict): Recorded trace.
        replay (dict): Replay trace.
        start (str): Replay start, "refreshed" or "stale".
        name (str): Demonstration name.
        experiment_b (dict): Arrays of Experiment B results.npz.

    Returns:
        dict: Largest error difference, goal-gap difference, and whether lift outcomes agree.
    """
    rows = experiment_b["demo"] == name
    saved = experiment_b[f"errors_{start}_recording_vs_absolute"][rows]
    # Experiment B columns: eef position, eef rotation, cube position.
    ours = achieved_errors(replay, recorded)[:, :3]
    i = list(experiment_b["demo_names"]).index(name)
    return {
        "errors": float(np.nanmax(np.abs(ours - saved))),
        "same_nan": bool(np.array_equal(np.isnan(ours), np.isnan(saved))),
        "goal_gap": float(np.abs(command_fidelity(replay) - experiment_b[f"goal_gap_absolute_{start}"][rows]).max()),
        "lift": bool(replay["success"][-1] == experiment_b[f"outcome_absolute_{start}_final"][i]),
    }


def check_geodesic(seed=0):
    """Compare geodesic angles with the known angles of random test rotations up to pi - 1e-6.

    Args:
        seed (int): Random seed.

    Returns:
        float: Largest absolute angle error in radians.
    """
    rng = np.random.default_rng(seed)
    angle = np.r_[rng.uniform(1e-3, np.pi - 1e-6, 2000), np.pi - np.logspace(-6, -2, 50)]
    axis = rng.normal(size=(len(angle), 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    rot = Rotation.from_rotvec(axis * angle[:, None]).as_matrix()
    base = Rotation.random(len(angle), random_state=seed).as_matrix()
    # Angle of base^T (base R) must equal the angle of R.
    return float(np.abs(geodesic(base, base @ rot) - angle).max())


def check_rotation(recorded, replay, abs_actions):
    """Check that stored rotation-vector jumps cause no orientation-error spikes.

    Args:
        recorded (dict): Recorded trace.
        replay (dict): Replay trace.
        abs_actions (np.ndarray): Stored absolute actions, shape (N, 7).

    Returns:
        dict: Jump steps, largest stored jump, largest true command rotation, largest e_R change, and largest slack over the bound.
    """
    jumps = np.flatnonzero(np.linalg.norm(np.diff(abs_actions[:, 3:6], axis=0), axis=1) > ROT_JUMP) + 1
    if not len(jumps):
        return {"jumps": [], "stored": 0.0, "true": 0.0, "change": 0.0, "slack": -np.inf}
    true = geodesic(replay["cmd_rot"][jumps - 1], replay["cmd_rot"][jumps])
    changes, slacks = [], []
    for trace in (recorded, replay):
        change = np.abs(tracking_errors(trace)[jumps, 1] - tracking_errors(trace)[jumps - 1, 1])
        # |e_R(t) - e_R(t-1)| <= d(R_{t-1}^cmd, R_t^cmd) + d(R_t^ach, R_{t+1}^ach); a branch artifact would break this by about pi.
        bound = true + geodesic(trace["ach_rot"][jumps], trace["ach_rot"][jumps + 1])
        changes.append(change)
        slacks.append(change - bound)
    return {
        "jumps": jumps.tolist(),
        "stored": float(np.linalg.norm(np.diff(abs_actions[:, 3:6], axis=0), axis=1)[jumps - 1].max()),
        "true": float(true.max()),
        "change": float(np.nanmax(np.concatenate(changes))),
        "slack": float(np.nanmax(np.concatenate(slacks))),
    }


def check_velocity_readout(env, demo, traces):
    """Compare stored site velocities with Jacobian x qvel at each stored state.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (h5py.Group): Demonstration that supplies the model.
        traces (list): Traces of this demonstration.

    Returns:
        float: Largest absolute difference over linear and angular components.
    """
    load_demo_model(env, demo)
    model, data = env.env.sim.model._model, env.env.sim.data._data
    site = env.env.robots[0].eef_site_id["right"]
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    worst = 0.0
    for trace in traces:
        for k in np.flatnonzero(trace["valid"]):
            env.reset_to({"states": trace["state"][k]})
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site)
            worst = max(worst, float(np.abs(jac_pos @ data.qvel - trace["ach_linvel"][k]).max()), float(np.abs(jac_rot @ data.qvel - trace["ach_angvel"][k]).max()))
    return worst


def check_demo(traces_file, source, name, experiment_b, env):
    """Run all checks on one demonstration.

    Args:
        traces_file (h5py.File): Traces file.
        source (h5py.Group): Original "data" group.
        name (str): Demonstration name.
        experiment_b (dict): Arrays of Experiment B results.npz.
        env (EnvRobosuite): Environment for the velocity readout check.

    Returns:
        dict: Check results.
    """
    recorded, _ = read_trace(traces_file[f"{name}/recorded"])
    replay, replay_attrs = read_trace(traces_file[f"{name}/replay_absolute"])
    with h5py.File(traces_file["meta"].attrs["absolute_file"], "r") as abs_file:
        abs_actions = abs_file[f"data/{name}/actions"][:]
    n = len(recorded["t_start"])
    obs = source[f"{name}/obs"]
    site_rot = Rotation.from_quat(obs["robot0_eef_quat_site"][:]).as_matrix()
    body_rot = Rotation.from_quat(obs["robot0_eef_quat"][:]).as_matrix()
    k0 = {name: bool(np.array_equal(recorded[name][0], replay[name][0])) for name in ("ach_pos", "ach_rot", "cube_pos", "finger_q")}
    return {
        "n": n,
        "start": replay_attrs["start"],
        "time": {"recorded": check_time(recorded, "recorded"), "replay": check_time(replay, "replay")},
        "goal_pos": float(np.linalg.norm(replay["goal_pos"] - replay["cmd_pos"], axis=1).max()),
        "goal_rot": float(angles(replay["goal_rot"], replay["cmd_rot"]).max()),
        "frame_pos": float(np.linalg.norm(recorded["ach_pos"][:n] - obs["robot0_eef_pos"][:], axis=1).max()),
        "frame_rot": float(angles(recorded["ach_rot"][:n], site_rot).max()),
        "body_rot": float(np.degrees(np.median(angles(recorded["ach_rot"][:n], body_rot)))),
        "same_start": k0,
        "complete": {"recorded": check_completeness(recorded), "replay": check_completeness(replay)},
        "velocity": check_velocity_readout(env, source[name], [recorded, replay]),
        "finite_difference": {"recorded": finite_difference_velocity(recorded), "replay": finite_difference_velocity(replay)},
        "repro": check_reproducibility(recorded, replay, replay_attrs["start"], name, experiment_b),
        "rotation": check_rotation(recorded, replay, abs_actions),
    }


def verdict(ok):
    """Return a pass or fail label.

    Args:
        ok (bool): Check outcome.

    Returns:
        str: "pass" or "FAIL".
    """
    return "pass" if ok else "FAIL"


def write_report(path, traces_path, meta, results, geodesic_error):
    """Write the validation report.

    Args:
        path (Path): Output report path.
        traces_path (Path): Traces file that was checked.
        meta (dict): File-level metadata.
        results (dict): Check results per demonstration.
        geodesic_error (float): Output of check_geodesic.

    Returns:
        bool: Whether every check passed.
    """
    rows, all_ok = [], True
    for name, r in results.items():
        tr, tp = r["time"]["recorded"], r["time"]["replay"]
        time_ok = all(x[k] <= TIME_TOL for x in (tr, tp) for k in ("interval", "start_times", "state_times")) and all(x["valid_states"] == x["expected_valid"] for x in (tr, tp))
        goal_ok = r["goal_pos"] <= GOAL_TOL and r["goal_rot"] <= GOAL_TOL
        frame_ok = r["frame_pos"] <= FRAME_TOL and r["frame_rot"] <= FRAME_ROT_TOL and all(r["same_start"].values())
        complete = r["complete"]
        complete_ok = not complete["recorded"]["problems"] and not complete["replay"]["problems"] and max(c["ortho"] for c in complete.values()) <= ORTHO_TOL
        vel_ok = r["velocity"] <= VELOCITY_TOL
        fd = r["finite_difference"]
        rp = r["repro"]
        repro_ok = rp["errors"] <= REPRO_TOL and rp["same_nan"] and rp["goal_gap"] <= REPRO_TOL and rp["lift"]
        ro = r["rotation"]
        rot_ok = ro["slack"] <= ROT_SLACK_TOL
        all_ok &= time_ok and goal_ok and frame_ok and complete_ok and vel_ok and repro_ok and rot_ok
        rows.append([
            name,
            r["n"],
            f"{verdict(time_ok)} ({max(tr['state_times'], tp['state_times']):.1e} s; states {tr['valid_states']}/{tp['valid_states']})",
            f"{verdict(goal_ok)} ({r['goal_pos']:.1e} m, {r['goal_rot']:.1e} rad)",
            f"{verdict(frame_ok)} ({r['frame_pos']:.1e} m, {r['frame_rot']:.1e} rad; body {r['body_rot']:.1f} deg)",
            f"{verdict(complete_ok)} (ortho {max(c['ortho'] for c in complete.values()):.1e})",
            f"{verdict(repro_ok)} ({rp['errors']:.1e}; lift {'same' if rp['lift'] else 'different'})",
            f"{verdict(rot_ok)} (" + (f"jumps at t = {ro['jumps']}, stored {ro['stored']:.2f} rad, true {ro['true']:.3f} rad, e_R change {ro['change']:.4f} rad, slack {ro['slack']:.4f} rad" if ro["jumps"] else "no jumps") + ")",
            f"{verdict(vel_ok)} ({r['velocity']:.1e})",
            f"lin {fd['recorded']['linear']:.3f}/{fd['replay']['linear']:.3f}, ang {fd['recorded']['angular']:.3f}/{fd['replay']['angular']:.3f}",
        ])
    lines = [
        "# Diagnostic trace validation",
        "",
        f"Traces: `{traces_path}`. Replay start: {', '.join(sorted({r['start'] for r in results.values()}))}.",
        f"Control frequency {meta['control_freq']} Hz, model timestep {meta['model_timestep']} s, {meta['substeps']} substeps per action.",
        f"Alignment: {meta['alignment']}",
        "",
        "| check | limit | what it compares |",
        "| --- | --- | --- |",
        f"| time | {TIME_TOL:g} s | interval lengths, action start times, state times against 0.05 k; valid states (recorded N / replay N + 1) |",
        f"| command fidelity | {GOAL_TOL:g} | replay controller goal against the commanded pose |",
        f"| frame | {FRAME_TOL:g} m, {FRAME_ROT_TOL:g} rad | recorded achieved pose against `obs/robot0_eef_pos` and `obs/robot0_eef_quat_site` (stored as float32); replay state 0 equal to recorded state 0. The body quaternion `robot0_eef_quat` is shown for contrast and must differ |",
        f"| completeness | {ORTHO_TOL:g} | shapes, finite valid rows, NaN invalid rows, rotation matrices orthonormal with determinant 1 |",
        f"| reproducibility | {REPRO_TOL:g} | replay vs recording errors (eef position, eef rotation, cube position), goal gaps, and final lift against Experiment B `results.npz` with the same start |",
        f"| rotation | slack {ROT_SLACK_TOL:g} rad | at steps where the stored rotation vector jumps by more than {ROT_JUMP:g} rad: the change of e_R (recorded and replay) minus its bound, the command rotation from t - 1 to t plus the achieved rotation from t to t + 1 |",
        f"| velocity | {VELOCITY_TOL:g} | stored grip-site linear and angular velocity against Jacobian x qvel at every stored state |",
        "| finite difference (information only) | none | median relative error of the interval-mean sampled velocity against finite differences of the poses (recorded/replay), on moving or turning steps |",
        "",
        table(["demo", "N", "time", "command fidelity", "frame", "completeness", "reproducibility", "rotation", "velocity", "finite difference (info)"], rows),
        "",
        f"Geodesic angle on 2050 test rotations up to pi - 1e-6: largest error {geodesic_error:.1e} rad (limit {GEODESIC_TOL:g}), {verdict(geodesic_error <= GEODESIC_TOL)}.",
        "",
        f"Overall: {verdict(all_ok and geodesic_error <= GEODESIC_TOL)}.",
        "",
    ]
    for name, r in results.items():
        for kind in ("recorded", "replay"):
            for problem in r["complete"][kind]["problems"]:
                lines.append(f"- {name} {kind}: {problem}")
    path.write_text("\n".join(lines))
    return all_ok and geodesic_error <= GEODESIC_TOL


def main():
    """Validate a traces file and write the report.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("docs/diagnostics/baseline_traces.hdf5"))
    parser.add_argument("--source", type=Path, default=Path("data/robomimic/datasets/lift/ph/low_dim_v15.hdf5"))
    parser.add_argument("--experiment-b", type=Path, default=Path("docs/continuous-replay/results.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/diagnostics"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.traces, "r") as traces_file, h5py.File(args.source, "r") as source_file:
        _, meta = read_trace(traces_file["meta"])
        names = demo_keys({k: None for k in traces_file if k != "meta"})
        experiment_b = dict(np.load(args.experiment_b))
        env = make_env(json.loads(meta["env_args"]), source_file[f"data/{names[0]}/obs"].keys())
        results = {name: check_demo(traces_file, source_file["data"], name, experiment_b, env) for name in names}
    ok = write_report(args.out_dir / "validation_report.md", args.traces, meta, results, check_geodesic())
    print(f"Report: {args.out_dir / 'validation_report.md'} ({verdict(ok)})")


if __name__ == "__main__":
    main()
