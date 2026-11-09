"""Compare one control step under delta and absolute actions from the same recorded states."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import arm_controller, demo_keys, load_demo_model, make_env, read_result, require
from inspect_lift_demo import table

# Slices of the flattened Lift state [time, qpos (16), qvel (15)].
STATE_BLOCKS = {
    "arm_q": (1, 8),
    "finger_q": (8, 10),
    "cube_q": (10, 17),
    "arm_v": (17, 24),
    "finger_v": (24, 26),
    "cube_v": (26, 32),
}
METRICS = ["eef_pos_m", "eef_rot_rad", *STATE_BLOCKS]
# Limits for delta vs absolute only, set after reviewing demo_0.
THRESHOLDS = {
    "eef_pos_m": 1e-8,
    "eef_rot_rad": 1e-8,
    "arm_q": 1e-8,
    "finger_q": 1e-8,
    "cube_q": 1e-8,
    "arm_v": 1e-6,
    "finger_v": 1e-6,
    "cube_v": 1e-6,
}


def one_step(env, state, action, refresh=True):
    """Restore a recorded state and run one control step.

    Args:
        env (EnvRobosuite): The environment wrapper.
        state (np.ndarray): Recorded flattened state, shape (32,).
        action (np.ndarray): Action, shape (7,).
        refresh (bool): Whether the controller rereads the restored robot pose.

    Returns:
        dict: Output of read_result.
    """
    env.reset_to({"states": state})
    if refresh:
        # After a model load, new_update stays False, so the first set_goal would use a pose from before reset_to.
        arm_controller(env).new_update = True
    env.step(action)
    return read_result(env)


def state_errors(state_a, state_b):
    """Return the largest absolute difference in each state block.

    Args:
        state_a (np.ndarray): First flattened state, shape (32,).
        state_b (np.ndarray): Second flattened state, shape (32,).

    Returns:
        list: One value per entry of STATE_BLOCKS.
    """
    return [float(np.abs(state_a[i:j] - state_b[i:j]).max()) for i, j in STATE_BLOCKS.values()]


def pose_errors(pos_a, rot_a, pos_b, rot_b):
    """Return the position distance and rotation angle between two poses.

    Args:
        pos_a (np.ndarray): First position, shape (3,).
        rot_a (np.ndarray): First rotation matrix, shape (3, 3).
        pos_b (np.ndarray): Second position, shape (3,).
        rot_b (np.ndarray): Second rotation matrix, shape (3, 3).

    Returns:
        list: Position distance in meters and rotation angle in radians.
    """
    return [float(np.linalg.norm(pos_a - pos_b)), float(Rotation.from_matrix(rot_a.T @ rot_b).magnitude())]


def compare_results(a, b):
    """Return all metrics between two step results.

    Args:
        a (dict): Output of read_result.
        b (dict): Output of read_result.

    Returns:
        list: One value per entry of METRICS.
    """
    return pose_errors(a["pos"], a["rot"], b["pos"], b["rot"]) + state_errors(a["state"], b["state"])


def compare_to_recording(result, demo_arrays, t):
    """Return all metrics between a step result and the recorded next moment.

    Args:
        result (dict): Output of read_result.
        demo_arrays (dict): Recorded states and next_obs pose arrays.
        t (int): Control step of the action.

    Returns:
        list: One value per entry of METRICS. State values are NaN at the final step.
    """
    recorded_rot = Rotation.from_quat(demo_arrays["next_site"][t]).as_matrix()
    pose = pose_errors(result["pos"], result["rot"], demo_arrays["next_pos"][t], recorded_rot)
    if t + 1 < len(demo_arrays["states"]):
        return pose + state_errors(result["state"], demo_arrays["states"][t + 1])
    # The file has no recorded state after the final action.
    return pose + [np.nan] * len(STATE_BLOCKS)


def run_demo(envs, source_demo, abs_demo, use_control):
    """Run Experiment A on one demonstration.

    Args:
        envs (dict): Environments named delta, absolute, control, stale_delta, and stale_absolute.
        source_demo (h5py.Group): Demonstration with delta actions.
        abs_demo (h5py.Group): Demonstration with absolute actions.
        use_control (bool): Whether to run the repeated delta control.

    Returns:
        dict: Per-step metric arrays, and one stale first-step record.
    """
    delta, absolute = source_demo["actions"][:], abs_demo["actions"][:]
    arrays = {
        "states": source_demo["states"][:],
        "next_pos": source_demo["next_obs/robot0_eef_pos"][:],
        "next_site": source_demo["next_obs/robot0_eef_quat_site"][:],
    }
    states = arrays["states"]
    require(np.array_equal(states, abs_demo["states"][:]), "the two files hold different states")
    for name, env in envs.items():
        if env is not None:
            load_demo_model(env, source_demo)

    # A fresh model load leaves these environments in the stale condition.
    stale_d = one_step(envs["stale_delta"], states[0], delta[0], refresh=False)
    stale_a = one_step(envs["stale_absolute"], states[0], absolute[0], refresh=False)
    stale = {
        "delta_vs_absolute": compare_results(stale_d, stale_a),
        "delta_vs_recording": compare_to_recording(stale_d, arrays, 0),
        "absolute_vs_recording": compare_to_recording(stale_a, arrays, 0),
    }

    rows = {key: [] for key in ("delta_vs_absolute", "delta_vs_recording", "absolute_vs_recording", "control")}
    # Steps run in order so the gripper's private command value builds up as it did in the recording.
    for t in range(len(delta)):
        d = one_step(envs["delta"], states[t], delta[t])
        a = one_step(envs["absolute"], states[t], absolute[t])
        rows["delta_vs_absolute"].append(compare_results(d, a))
        rows["delta_vs_recording"].append(compare_to_recording(d, arrays, t))
        rows["absolute_vs_recording"].append(compare_to_recording(a, arrays, t))
        if use_control:
            rows["control"].append(compare_results(d, one_step(envs["control"], states[t], delta[t])))
    return {"rows": {k: np.array(v) for k, v in rows.items()}, "stale": stale}


def summary_rows(values, steps):
    """Build table rows with median, 95th percentile, maximum, and worst step per metric.

    Args:
        values (np.ndarray): Metric values, shape (M, len(METRICS)).
        steps (list): Label of each row, length M.

    Returns:
        list: One row per metric.
    """
    rows = []
    for i, name in enumerate(METRICS):
        column = values[:, i]
        if np.isnan(column).all():
            continue
        worst = int(np.nanargmax(column))
        rows.append((name, f"{np.nanmedian(column):.2e}", f"{np.nanpercentile(column, 95):.2e}", f"{column[worst]:.2e}", steps[worst]))
    return rows


def threshold_rows(values, labels):
    """Compare each metric with its limit and list the steps that exceed it.

    Args:
        values (np.ndarray): Delta-vs-absolute values, shape (M, len(METRICS)).
        labels (list): Label of each row, length M.

    Returns:
        tuple: Table rows, and True when no step exceeds a limit.
    """
    rows, passed = [], True
    for i, name in enumerate(METRICS):
        over = np.flatnonzero(values[:, i] > THRESHOLDS[name])
        passed = passed and over.size == 0
        rows.append((name, f"{THRESHOLDS[name]:.0e}", f"{values[:, i].max():.2e}", over.size, ", ".join(labels[j] for j in over[:5]) or "none"))
    return rows, passed


def write_report(path, names, results, first_closing, env_settings):
    """Write the Markdown Experiment A report.

    Args:
        path (Path): Output report path.
        names (list): Demonstration names.
        results (list): Output of run_demo per demonstration.
        first_closing (list): First closing command step per demonstration.
        env_settings (dict): Live settings of the two controllers.

    Returns:
        None: Nothing.
    """
    labels = [f"{name} t={t}" for name, r in zip(names, results) for t in range(len(r["rows"]["delta_vs_absolute"]))]
    stack = lambda key: np.concatenate([r["rows"][key] for r in results])
    head = ["Metric", "Median", "95th pct", "Max", "Worst step"]
    is_first = np.array([label.endswith(" t=0") for label in labels])
    later_labels = [label for label, first in zip(labels, is_first) if not first]
    delta_vs_absolute = stack("delta_vs_absolute")
    limits, passed = threshold_rows(delta_vs_absolute, labels)

    sections = [
        "# Experiment A: one-step action equivalence\n\n"
        f"Demonstrations: {len(names)}. Control steps: {len(labels)}. Each step restores the recorded state, refreshes the controller, runs 25 physics steps, and then reads the state after `sim.forward()`.",
        "## Controllers\n\n" + table(["Setting", "Delta environment", "Absolute environment"], [(k, *v) for k, v in env_settings.items()]),
        f"## Result\n\n**{'All steps are within the limits.' if passed else 'Some steps exceed the limits.'}** The limits apply to delta vs absolute only. Comparison B is report-only.\n\n"
        + table(["Metric", "Limit", "Max", "Steps over limit", "First steps over limit"], limits),
        "## A. Delta vs absolute (primary)\n\nAll steps, including t = 0.\n\n" + table(head, summary_rows(delta_vs_absolute, labels)),
        "Steps after t = 0.\n\n" + table(head, summary_rows(delta_vs_absolute[~is_first], later_labels)),
    ]
    if results[0]["rows"]["control"].size:
        sections.append("## Control: delta vs a second delta environment\n\n" + table(head, summary_rows(stack("control"), labels)))
    sections += [
        "## B. Each controller vs the recording\n\nStep t is compared with recorded step t+1. Pose rows use `next_obs`, and state rows use `states[t+1]`, so state rows have no value at the final step.\n\n"
        "Delta\n\n" + table(head, summary_rows(stack("delta_vs_recording"), labels)),
        "Absolute\n\n" + table(head, summary_rows(stack("absolute_vs_recording"), labels)),
    ]
    finals = [(name, f"{r['rows']['delta_vs_recording'][-1, 0]:.2e}", f"{r['rows']['absolute_vs_recording'][-1, 0]:.2e}", f"{r['rows']['delta_vs_recording'][-1, 1]:.2e}", f"{r['rows']['absolute_vs_recording'][-1, 1]:.2e}") for name, r in zip(names, results)]
    if len(finals) <= 5:
        sections.append("## Final step vs `next_obs`\n\n" + table(["Demo", "Delta pos (m)", "Absolute pos (m)", "Delta rot (rad)", "Absolute rot (rad)"], finals))
    else:
        final_pos = np.array([r["rows"]["delta_vs_recording"][-1, 0] for r in results])
        final_rot = np.array([r["rows"]["delta_vs_recording"][-1, 1] for r in results])
        before_final = np.concatenate([r["rows"]["delta_vs_recording"][:-1, 0] for r in results])
        sections.append(
            "## Final step vs `next_obs`\n\nThe final `next_obs` row has no matching `obs` row, so it is reported apart from the other steps.\n\n"
            + table(
                ["Measure", "Median", "Max"],
                [
                    ("Final step, delta EEF position (m)", f"{np.median(final_pos):.2e}", f"{final_pos.max():.2e}"),
                    ("Final step, delta EEF rotation (rad)", f"{np.median(final_rot):.2e}", f"{final_rot.max():.2e}"),
                    ("All earlier steps, delta EEF position (m)", f"{np.median(before_final):.2e}", f"{before_final.max():.2e}"),
                ],
            )
        )

    first_rows = []
    for key, stale_key, label in (
        ("delta_vs_absolute", "delta_vs_absolute", "Delta vs absolute"),
        ("delta_vs_recording", "delta_vs_recording", "Delta vs recording"),
        ("absolute_vs_recording", "absolute_vs_recording", "Absolute vs recording"),
    ):
        refreshed = np.array([r["rows"][key][0] for r in results])
        stale = np.array([r["stale"][stale_key] for r in results])
        for j in (0, METRICS.index("arm_v")):
            first_rows.append((label, METRICS[j], f"{np.median(refreshed[:, j]):.2e}", f"{refreshed[:, j].max():.2e}", f"{np.median(stale[:, j]):.2e}", f"{stale[:, j].max():.2e}"))
    sections.append(
        "## First step, t = 0\n\nStale means the first step after a model load, with no controller refresh.\n\n"
        + table(["Comparison", "Metric", "Refreshed median", "Refreshed max", "Stale median", "Stale max"], first_rows)
    )

    # A step can only be blamed on the converter if the controllers disagree there.
    gaps = stack("delta_vs_recording")
    # Final steps have no recorded state, so they rank last here. They have their own section.
    full_state = np.nan_to_num(gaps[:, 2:], nan=-1.0).max(axis=1)
    order = np.argsort(full_state)[::-1][:8]
    anomaly_rows = [
        (labels[i], f"{full_state[i]:.2e}", METRICS[2 + int(np.nanargmax(gaps[i, 2:]))], f"{np.nanmax(delta_vs_absolute[i, 2:]):.2e}", f"{stack('absolute_vs_recording')[i, 0]:.2e}")
        for i in order
    ]
    sections.append(
        "## Largest differences from the recording\n\nThe delta-vs-absolute column shows whether the two controllers disagree at the same step.\n\n"
        + table(["Step", "Delta vs recording", "Largest block", "Delta vs absolute", "Absolute EEF pos error (m)"], anomaly_rows)
    )
    if len(names) <= 5:
        sections.append("First closing command: " + ", ".join(f"{n} t={c}" for n, c in zip(names, first_closing)) + ".")
    path.write_text("\n\n".join(sections) + "\n")
    print("All steps are within the limits." if passed else "Some steps exceed the limits.")


def main():
    """Run Experiment A and write the report and per-step results.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/lift/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--absolute", type=Path, default=root / "low_dim_abs.hdf5")
    parser.add_argument("--demos", nargs="+", default=["demo_0"], help='Demonstration names, or "all"')
    parser.add_argument("--control", action="store_true", help="Also run a repeated delta environment")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/one-step-equivalence"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source, "r") as source_file, h5py.File(args.absolute, "r") as abs_file:
        source, output = source_file["data"], abs_file["data"]
        names = demo_keys(source) if args.demos == ["all"] else args.demos
        obs_keys = source[f"{names[0]}/obs"].keys()
        delta_args = json.loads(output.attrs["env_args_delta"])
        abs_args = json.loads(output.attrs["env_args"])
        envs = {
            "delta": make_env(delta_args, obs_keys),
            "absolute": make_env(abs_args, obs_keys),
            "control": make_env(delta_args, obs_keys) if args.control else None,
            "stale_delta": make_env(delta_args, obs_keys),
            "stale_absolute": make_env(abs_args, obs_keys),
        }
        require(arm_controller(envs["delta"]).input_type == "delta", "delta environment is not delta")
        require(arm_controller(envs["absolute"]).input_type == "absolute", "absolute environment is not absolute")

        results, first_closing = [], []
        for i, name in enumerate(names):
            results.append(run_demo(envs, source[name], output[name], args.control))
            first_closing.append(int(np.argmax(source[name]["actions"][:, 6] > 0)))
            if i % 25 == 0:
                print(f"Finished {name}", flush=True)

    settings = {}
    for field in ("input_type", "input_ref_frame", "ref_name", "goal_update_mode"):
        attribute = "_goal_update_mode" if field == "goal_update_mode" else field
        settings[field] = (getattr(arm_controller(envs["delta"]), attribute), getattr(arm_controller(envs["absolute"]), attribute))
    for field, reader in (("kp", lambda c: c.kp[0]), ("kd", lambda c: round(float(c.kd[0]), 4))):
        settings[field] = (reader(arm_controller(envs["delta"])), reader(arm_controller(envs["absolute"])))
    settings["control_freq (Hz)"] = (envs["delta"].env.control_freq, envs["absolute"].env.control_freq)
    settings["model timestep (s)"] = (float(envs["delta"].env.sim.model.opt.timestep), float(envs["absolute"].env.sim.model.opt.timestep))

    suffix = "" if args.demos == ["all"] else "_" + "_".join(names)
    write_report(args.out_dir / f"report{suffix}.md", names, results, first_closing, settings)
    np.savez(
        args.out_dir / f"results{suffix}.npz",
        metrics=np.array(METRICS),
        demo=np.concatenate([[name] * len(r["rows"]["delta_vs_absolute"]) for name, r in zip(names, results)]),
        t=np.concatenate([np.arange(len(r["rows"]["delta_vs_absolute"])) for r in results]),
        **{key: np.concatenate([r["rows"][key] for r in results]) for key in ("delta_vs_absolute", "delta_vs_recording", "absolute_vs_recording", "control")},
        **{f"stale_{key}": np.array([r["stale"][key] for r in results]) for key in results[0]["stale"]},
    )
    print(f"Report: {args.out_dir / f'report{suffix}.md'}")


if __name__ == "__main__":
    main()
