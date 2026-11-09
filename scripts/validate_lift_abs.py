"""Validate the converted Lift dataset against the original and write a report."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import arm_controller, demo_keys, make_env, require
from inspect_lift_demo import table

POS_TOL = 1e-9
ROT_TOL = 1e-5
POS_SCALE = 0.05
ROT_SCALE = 0.5
JUMP_THRESHOLD = 1.0


def rotation_error(goal_a, goal_b):
    """Return the rotation angle between two sets of rotations.

    Args:
        goal_a (Rotation): First rotations.
        goal_b (Rotation): Second rotations.

    Returns:
        np.ndarray: Angles in radians.
    """
    return (goal_a.inv() * goal_b).magnitude()


def check_preserved(source, output):
    """Require that everything except the pose actions and metadata is exactly unchanged.

    Args:
        source (h5py.Group): Original "data" group.
        output (h5py.Group): Converted "data" group.

    Returns:
        int: Number of arrays compared.
    """
    require(demo_keys(source) == demo_keys(output), "demonstration names differ")
    compared = 0
    for key in demo_keys(source):
        a, b = source[key], output[key]
        require(set(a.keys()) == set(b.keys()), f"{key} has different groups")
        require(a["actions"].shape == b["actions"].shape, f"{key} actions shape changed")
        require(np.array_equal(a["actions"][:, 6], b["actions"][:, 6]), f"{key} gripper commands changed")
        names = ["states", "rewards", "dones"]
        names += [f"obs/{k}" for k in a["obs"]] + [f"next_obs/{k}" for k in a["next_obs"]]
        for name in names:
            require(a[name].dtype == b[name].dtype, f"{key}/{name} dtype changed")
            require(np.array_equal(a[name][:], b[name][:]), f"{key}/{name} changed")
        require(set(a.attrs) == set(b.attrs), f"{key} attributes changed")
        for attr in a.attrs:
            require(a.attrs[attr] == b.attrs[attr], f"{key} attribute {attr} changed")
        compared += len(names) + 1
    return compared


def check_goals(source, output):
    """Compare goals with the controller-free formula and collect plausibility numbers.

    Args:
        source (h5py.Group): Original "data" group.
        output (h5py.Group): Converted "data" group.

    Returns:
        dict: Largest errors, goal ranges, and rotation-vector jump counts.
    """
    stats = {"pos_error": 0.0, "rot_error": 0.0, "goal_distance": 0.0, "goal_rotation": 0.0}
    goals, jumps, jump_demos, norms = [], 0, 0, []
    for key in demo_keys(source):
        delta, absolute = source[key]["actions"][:], output[key]["actions"][:]
        require(np.isfinite(absolute).all(), f"{key} has non-finite actions")
        position = source[key]["obs/robot0_eef_pos"][:]
        site = Rotation.from_quat(source[key]["obs/robot0_eef_quat_site"][:])
        expected_pos = position + POS_SCALE * np.clip(delta[:, :3], -1, 1)
        expected_rot = Rotation.from_rotvec(ROT_SCALE * np.clip(delta[:, 3:6], -1, 1)) * site
        goal_rot = Rotation.from_rotvec(absolute[:, 3:6])
        stats["pos_error"] = max(stats["pos_error"], float(np.linalg.norm(absolute[:, :3] - expected_pos, axis=1).max()))
        stats["rot_error"] = max(stats["rot_error"], float(rotation_error(goal_rot, expected_rot).max()))
        stats["goal_distance"] = max(stats["goal_distance"], float(np.linalg.norm(absolute[:, :3] - position, axis=1).max()))
        stats["goal_rotation"] = max(stats["goal_rotation"], float(rotation_error(site, goal_rot).max()))
        count = int((np.linalg.norm(np.diff(absolute[:, 3:6], axis=0), axis=1) > JUMP_THRESHOLD).sum())
        jumps += count
        jump_demos += count > 0
        goals.append(absolute[:, :6])
        norms.append(np.linalg.norm(absolute[:, 3:6], axis=1))
    goals = np.concatenate(goals)
    norms = np.concatenate(norms)
    stats.update(
        goal_min=goals.min(axis=0),
        goal_max=goals.max(axis=0),
        jumps=jumps,
        jump_demos=jump_demos,
        rotvec_norm_min=float(norms.min()),
        rotvec_norm_median=float(np.median(norms)),
        rotvec_norm_max=float(norms.max()),
        outside_unit_range=float((np.abs(goals) > 1.0).mean()),
    )
    require(stats["pos_error"] <= POS_TOL, f"position differs from the formula by {stats['pos_error']:.2e} m")
    require(stats["rot_error"] <= ROT_TOL, f"rotation differs from the formula by {stats['rot_error']:.2e} rad")
    require(stats["goal_distance"] <= POS_SCALE * np.sqrt(3) + POS_TOL, "a goal is too far from the current position")
    require(stats["goal_rotation"] <= ROT_SCALE * np.sqrt(3) + ROT_TOL, "a goal rotates too far from the current orientation")
    require(stats["rotvec_norm_max"] <= np.pi + ROT_TOL, "a rotation vector is longer than pi")
    return stats


def check_controllers(source, output):
    """Require that the file's absolute controller and the delta controller set the same goal.

    Args:
        source (h5py.Group): Original "data" group.
        output (h5py.Group): Converted "data" group.

    Returns:
        dict: Live controller settings, largest goal differences, and steps compared.
    """
    keys = demo_keys(source)
    obs_keys = source[f"{keys[0]}/obs"].keys()
    # Build each environment from the converted file's own metadata, as later code will.
    delta_env = make_env(json.loads(output.attrs["env_args_delta"]), obs_keys)
    abs_env = make_env(json.loads(output.attrs["env_args"]), obs_keys)
    delta_controller, abs_controller = arm_controller(delta_env), arm_controller(abs_env)
    require(delta_controller.input_type == "delta", "env_args_delta does not create a delta controller")
    require(abs_controller.input_type == "absolute", "env_args does not create an absolute controller")
    require(abs_controller.input_ref_frame == "world", "absolute controller does not use the world frame")
    require("grip_site" in abs_controller.ref_name, "absolute controller does not use the grip site")

    robot = abs_env.env.robots[0]
    pos_diff, rot_diff, steps = 0.0, 0.0, 0
    for key in keys:
        states, delta, absolute = source[key]["states"][:], source[key]["actions"][:], output[key]["actions"][:]
        for t in range(len(states)):
            delta_env.reset_to({"states": states[t]})
            delta_controller.update(force=True)
            delta_controller.set_goal(delta[t, :6])
            abs_env.reset_to({"states": states[t]})
            # robot.control tests the full 7-number action path but runs no physics step.
            robot.control(absolute[t], policy_step=True)
            pos_diff = max(pos_diff, float(np.linalg.norm(abs_controller.goal_pos - delta_controller.goal_pos)))
            angle = rotation_error(Rotation.from_matrix(abs_controller.goal_ori), Rotation.from_matrix(delta_controller.goal_ori))
            rot_diff = max(rot_diff, float(angle))
            steps += 1
    require(pos_diff <= POS_TOL, f"controller goal positions differ by {pos_diff:.2e} m")
    require(rot_diff <= ROT_TOL, f"controller goal orientations differ by {rot_diff:.2e} rad")
    return {
        "input_type": abs_controller.input_type,
        "input_ref_frame": abs_controller.input_ref_frame,
        "ref_name": abs_controller.ref_name,
        "goal_update_mode": abs_controller._goal_update_mode,
        "pos_diff": pos_diff,
        "rot_diff": rot_diff,
        "steps": steps,
    }


def write_report(path, summary, goals, controllers, first_rows, delta_rows, env_args, convention):
    """Write the Markdown conversion report.

    Args:
        path (Path): Output report path.
        summary (dict): Dataset integrity values.
        goals (dict): Output of check_goals.
        controllers (dict): Output of check_controllers.
        first_rows (np.ndarray): First absolute actions of demo_0, shape (M, 7).
        delta_rows (np.ndarray): First delta actions of demo_0, shape (M, 7).
        env_args (dict): Metadata stored in the converted file.
        convention (dict): The action_convention attribute.

    Returns:
        None: Nothing.
    """
    axes = ["x", "y", "z", "rx", "ry", "rz"]
    fmt = lambda row: [f"{value:.4f}" for value in row]
    arm = env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    sections = [
        "# Step 2 conversion report: Lift PH low-dim\n\n"
        "All hard checks passed. This report covers structure and representation. It does not claim that absolute actions reproduce the original motion, which Step 3 tests.",
        "## Dataset integrity\n\n" + table(["Check", "Result"], list(summary.items())),
        "## Action convention\n\n" + table(["Field", "Value"], list(convention.items())),
        "## Absolute-controller configuration\n\n"
        + table(["Setting", "Value"], [("Stored type", arm["type"]), ("Stored input_type", arm["input_type"]), ("Stored input_ref_frame", arm["input_ref_frame"]), ("control_delta present", "control_delta" in arm)] + [(f"Live {k}", controllers[k]) for k in ("input_type", "input_ref_frame", "ref_name", "goal_update_mode")])
        + "\n\nThe original delta metadata is kept in the `env_args_delta` attribute.",
        "## Goal checks\n\n"
        + table(
            ["Measure", "Largest value", "Limit"],
            [
                ("Position vs controller-free formula (m)", f"{goals['pos_error']:.2e}", f"{POS_TOL:.0e}"),
                ("Rotation vs controller-free formula (rad)", f"{goals['rot_error']:.2e}", f"{ROT_TOL:.0e}"),
                ("Absolute vs delta controller goal position (m)", f"{controllers['pos_diff']:.2e}", f"{POS_TOL:.0e}"),
                ("Absolute vs delta controller goal rotation (rad)", f"{controllers['rot_diff']:.2e}", f"{ROT_TOL:.0e}"),
                ("Goal distance from current position (m)", f"{goals['goal_distance']:.4f}", f"{POS_SCALE * np.sqrt(3):.4f}"),
                ("Goal rotation from current orientation (rad)", f"{goals['goal_rotation']:.4f}", f"{ROT_SCALE * np.sqrt(3):.4f}"),
            ],
        )
        + f"\n\nThe controller comparison covered {controllers['steps']} steps in all demonstrations.",
        "## Goal ranges\n\n" + table(["Channel", "Min", "Max"], [(a, f"{low:.3f}", f"{high:.3f}") for a, low, high in zip(axes, goals["goal_min"], goals["goal_max"])]),
        "## First actions of demo_0\n\nDelta\n\n"
        + table(["t", *axes, "gripper"], [[t, *fmt(row)] for t, row in enumerate(delta_rows)])
        + "\n\nAbsolute\n\n"
        + table(["t", *axes, "gripper"], [[t, *fmt(row)] for t, row in enumerate(first_rows)]),
        "## Warnings\n\n"
        f"- Rotation vectors jump between steps. There are {goals['jumps']} jumps above {JUMP_THRESHOLD} rad in {goals['jump_demos']} demonstrations, because the rotation angle is near pi (min {goals['rotvec_norm_min']:.3f}, median {goals['rotvec_norm_median']:.3f}, max {goals['rotvec_norm_max']:.3f} rad). "
        "The rotations are valid, but later code must not average or fit these three numbers directly.\n"
        f"- {goals['outside_unit_range'] * 100:.1f}% of pose values lie outside [-1, 1]. robomimic's docs expect normalized actions, and its `get_dataset_info.py` check can reject this file.",
    ]
    path.write_text("\n\n".join(sections) + "\n")


def main():
    """Run all validation checks and write the report.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/lift/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--output", type=Path, default=root / "low_dim_abs.hdf5")
    parser.add_argument("--versions", type=Path, default=Path("docs/sim-and-dataset-inspection/env_versions.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/delta-to-absolute-conversion"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
    step1_sha256 = json.loads(args.versions.read_text())["dataset"]["sha256"]
    require(source_sha256 == step1_sha256, "the original dataset changed since Step 1")
    require(args.output.stat().st_mode & 0o222 == 0, "the output file is writable")

    with h5py.File(args.source, "r") as source_file, h5py.File(args.output, "r") as output_file:
        source, output = source_file["data"], output_file["data"]
        require(len(output) == 200, f"expected 200 demonstrations, found {len(output)}")
        require(output["demo_0/actions"].shape == (59, 7), "demo_0/actions is not (59, 7)")
        convention = json.loads(output.attrs["action_convention"])
        require(convention["source_sha256"] == source_sha256, "action_convention names a different source")
        compared = check_preserved(source, output)
        goals = check_goals(source, output)
        controllers = check_controllers(source, output)
        summary = {
            "Demonstrations": len(output),
            "demo_0/actions shape": output["demo_0/actions"].shape,
            "Total action rows": sum(len(output[k]["actions"]) for k in demo_keys(output)),
            "Arrays compared, all exactly equal": compared,
            "Compared arrays": "gripper commands, states, rewards, dones, obs, next_obs",
            "Original SHA-256, same as Step 1": f"`{source_sha256}`",
            "Output SHA-256": f"`{hashlib.sha256(args.output.read_bytes()).hexdigest()}`",
            "Output read-only": True,
        }
        write_report(
            args.out_dir / "report.md",
            summary,
            goals,
            controllers,
            output["demo_0/actions"][:3],
            source["demo_0/actions"][:3],
            json.loads(output.attrs["env_args"]),
            convention,
        )
    print(f"All hard checks passed. Report: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
