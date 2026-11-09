"""Check the Lift simulator setup and inspect one demonstration without changing the dataset."""

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import arm_controller, require

PACKAGES = ["numpy", "numba", "mujoco", "mink", "scipy", "h5py", "torch", "torchvision", "robosuite", "robomimic"]
QUAT_NORM_TOL = 1e-5


def collect_versions(dataset_path):
    """Import each package and record its version, commit, and the dataset hash.

    Args:
        dataset_path (Path): Path of the HDF5 dataset.

    Returns:
        dict: Python version, package records, and dataset SHA-256.
    """
    packages = {}
    for name in PACKAGES:
        # Importing directly exposes failures that robosuite's optional imports hide.
        importlib.import_module(name)
        direct_url = metadata.distribution(name).read_text("direct_url.json")
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None
        packages[name] = {"version": metadata.version(name), "commit": commit}
    return {
        "python": platform.python_version(),
        "packages": packages,
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
    }


def load_demo(dataset_path, demo_key):
    """Load one demonstration and the dataset metadata.

    Args:
        dataset_path (Path): Path of the HDF5 dataset.
        demo_key (str): Demonstration group name, such as "demo_0".

    Returns:
        dict: Arrays, observation dicts, env_args, and demonstration attributes.
    """
    with h5py.File(dataset_path, "r") as f:
        demo = f[f"data/{demo_key}"]
        return {
            "states": demo["states"][:],
            "actions": demo["actions"][:],
            "rewards": demo["rewards"][:],
            "dones": demo["dones"][:],
            "obs": {k: v[:] for k, v in demo["obs"].items()},
            "next_obs": {k: v[:] for k, v in demo["next_obs"].items()},
            "env_args": json.loads(f["data"].attrs["env_args"]),
            "model_file": demo.attrs.get("model_file"),
            "ep_meta": demo.attrs.get("ep_meta"),
            "attrs": sorted(demo.attrs.keys()),
            "num_demos": len(f["data"]),
        }


def check_arrays(demo):
    """Apply the hard shape, finiteness, and unit-norm checks.

    Args:
        demo (dict): Output of load_demo.

    Returns:
        int: Number of recorded actions N.
    """
    n = demo["actions"].shape[0]
    require(demo["states"].ndim == 2 and demo["states"].shape[0] == n, "states is not (N, D_s)")
    require(demo["actions"].shape == (n, 7), "actions is not (N, 7)")
    require(np.isfinite(demo["states"]).all(), "states has non-finite values")
    require(np.isfinite(demo["actions"]).all(), "actions has non-finite values")
    require(demo["obs"]["robot0_eef_pos"].shape == (n, 3), "robot0_eef_pos is not (N, 3)")
    require(demo["obs"]["robot0_eef_quat"].shape == (n, 4), "robot0_eef_quat is not (N, 4)")
    require(set(demo["obs"]) == set(demo["next_obs"]), "obs and next_obs have different keys")
    for key, value in demo["obs"].items():
        require(value.shape == demo["next_obs"][key].shape, f"next_obs/{key} shape differs from obs/{key}")
        require(value.shape[0] == n, f"obs/{key} does not have N rows")
        require(np.isfinite(value).all(), f"obs/{key} has non-finite values")
        require(np.isfinite(demo["next_obs"][key]).all(), f"next_obs/{key} has non-finite values")
    for key in ("robot0_eef_quat", "robot0_eef_quat_site"):
        if key in demo["obs"]:
            error = np.abs(np.linalg.norm(demo["obs"][key], axis=1) - 1.0).max()
            require(error < QUAT_NORM_TOL, f"{key} is not unit norm, error {error:.2e}")
    return n


def make_env(demo):
    """Create the environment from the stored dataset metadata.

    Args:
        demo (dict): Output of load_demo.

    Returns:
        EnvRobosuite: The robomimic environment wrapper.
    """
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    # EnvRobosuite.get_observation reads this table, so it must exist first.
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(demo["obs"]), "rgb": []}})
    return EnvUtils.create_env_from_metadata(demo["env_args"], render=False, render_offscreen=False, use_image_obs=False)


def live_settings(env):
    """Record the controller and timing settings that the simulator really uses.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        dict: Live settings as JSON-serializable values.
    """
    base = env.env
    controller = arm_controller(env)
    return {
        "env": type(base).__name__,
        "robot": base.robots[0].name,
        "controller_class": type(controller).__name__,
        "controller_name": controller.name,
        "input_type": controller.input_type,
        "input_ref_frame": controller.input_ref_frame,
        "goal_update_mode": controller._goal_update_mode,
        "ref_name": controller.ref_name,
        "interpolator_pos": type(controller.interpolator_pos).__name__,
        "input_max": np.asarray(controller.input_max).tolist(),
        "input_min": np.asarray(controller.input_min).tolist(),
        "output_max": np.asarray(controller.output_max).tolist(),
        "output_min": np.asarray(controller.output_min).tolist(),
        "kp": np.asarray(controller.kp).tolist(),
        "kd": np.asarray(controller.kd).tolist(),
        "control_freq_hz": base.control_freq,
        "control_timestep_s": base.control_timestep,
        "model_timestep_s": float(base.sim.model.opt.timestep),
        "sim_steps_per_control": base.control_timestep / float(base.sim.model.opt.timestep),
        "horizon": base.horizon,
        "action_dim": env.action_dimension,
        "nq": int(base.sim.model.nq),
        "nv": int(base.sim.model.nv),
    }


def zero_action_step(env):
    """Apply one zero action after a reset and measure the motion.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        dict: End-effector and gripper change after the step.
    """
    before = env.reset()
    after = env.step(np.zeros(env.action_dimension))[0]
    require(all(np.isfinite(v).all() for v in after.values()), "zero-action step gave non-finite observations")
    rotation = Rotation.from_quat(before["robot0_eef_quat"]).inv() * Rotation.from_quat(after["robot0_eef_quat"])
    return {
        "eef_displacement_m": float(np.linalg.norm(after["robot0_eef_pos"] - before["robot0_eef_pos"])),
        "eef_rotation_deg": float(np.degrees(rotation.magnitude())),
        "gripper_qpos_change": float(np.abs(after["robot0_gripper_qpos"] - before["robot0_gripper_qpos"]).max()),
    }


def restore_model(env, demo):
    """Load the demonstration's MuJoCo model and first state.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (dict): Output of load_demo.

    Returns:
        None: Nothing.
    """
    state = {"states": demo["states"][0], "model": demo["model_file"]}
    if demo["ep_meta"] is not None:
        state["ep_meta"] = demo["ep_meta"]
    env.reset_to(state)


def compare_reset_observations(env, demo):
    """Compare stored observations with observations rebuilt from stored states.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (dict): Output of load_demo.

    Returns:
        dict: Max absolute error per key for the first and second reads, and per object component.
    """
    base = env.env
    first, second, components = {}, {}, []
    for t, state in enumerate(demo["states"]):
        # LieSpline-DP reads twice after a reset, so measure whether the second read differs.
        first_read = env.reset_to({"states": state})
        second_read = env.get_observation()
        for key, stored in demo["obs"].items():
            first[key] = max(first.get(key, 0.0), float(np.abs(stored[t] - first_read[key]).max()))
            second[key] = max(second.get(key, 0.0), float(np.abs(stored[t] - second_read[key]).max()))
        if t == 0:
            names = [k for k, o in base._observables.items() if o.modality == "object" and o.is_enabled() and o.is_active()]
        raw = base._get_observations(force_update=True)
        parts = [np.atleast_1d(raw[name]) for name in names]
        offset = 0
        for i, (name, part) in enumerate(zip(names, parts)):
            stored_part = demo["obs"]["object"][t, offset : offset + part.size]
            error = float(np.abs(stored_part - part).max())
            if t == 0:
                components.append({"name": name, "dim": int(part.size), "max_error": error})
            else:
                components[i]["max_error"] = max(components[i]["max_error"], error)
            offset += part.size
    return {"first_read": first, "second_read": second, "object_components": components}


def compare_frames(env, demo):
    """Measure the frame offset between the body and site orientations, and the controller frame.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (dict): Output of load_demo.

    Returns:
        dict: Offset size, its variation over the demonstration, and the controller comparison.
    """
    obs = demo["obs"]
    if "robot0_eef_quat_site" not in obs:
        return {"site_key_stored": False}
    body = Rotation.from_quat(obs["robot0_eef_quat"])
    site = Rotation.from_quat(obs["robot0_eef_quat_site"])
    offset = body.inv() * site
    variation = offset[0].inv() * offset

    t = len(demo["states"]) // 2
    env.reset_to({"states": demo["states"][t]})
    controller = arm_controller(env)
    controller.update(force=True)
    ref = Rotation.from_matrix(controller.ref_ori_mat)
    return {
        "site_key_stored": True,
        "offset_angle_deg": float(np.degrees(offset[0].magnitude())),
        "offset_rotvec_rad": offset[0].as_rotvec().tolist(),
        "offset_variation_max_deg": float(np.degrees(variation.magnitude().max())),
        "probe_t": t,
        "controller_vs_site_deg": float(np.degrees((ref.inv() * site[t]).magnitude())),
        "controller_vs_body_deg": float(np.degrees((ref.inv() * body[t]).magnitude())),
        "controller_vs_eef_pos_m": float(np.linalg.norm(controller.ref_pos - obs["robot0_eef_pos"][t])),
    }


def single_step_replay(env, demo, n):
    """Step from stored states with the matching and neighboring actions.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (dict): Output of load_demo.
        n (int): Number of recorded actions.

    Returns:
        list: One record per tested t, with errors per candidate action index.
    """
    records = []
    nq = env.env.sim.model.nq
    controller = arm_controller(env)
    for t in (0, 1, n // 4, n // 2, 3 * n // 4, n - 2):
        record = {"t": t, "candidates": {}}
        for offset in (-1, 0, 1):
            if not 0 <= t + offset < n:
                continue
            env.reset_to({"states": demo["states"][t]})
            step_obs = env.step(demo["actions"][t + offset])[0]
            # After mj_step, site poses describe the qpos before the last integration. Stored observations
            # came from a forward call, so recompute kinematics before comparing with them.
            unforwarded_obs = env.get_observation()
            env.env.sim.forward()
            obs = env.get_observation()
            stored = {k: v[t + 1] for k, v in demo["obs"].items()}
            # The flattened state is [time, qpos, qvel]. Time is not part of the dynamics error.
            state_error = np.abs(env.get_state()["states"] - demo["states"][t + 1])
            rotation = Rotation.from_quat(obs["robot0_eef_quat"]).inv() * Rotation.from_quat(stored["robot0_eef_quat"])
            pos_error = lambda o: float(np.linalg.norm(o["robot0_eef_pos"] - stored["robot0_eef_pos"]))
            record["candidates"][offset] = {
                "arm_qpos_max_abs": float(state_error[1:][controller.qpos_index].max()),
                "qvel_max_abs": float(state_error[1 + nq :].max()),
                "eef_pos_m": pos_error(obs),
                "eef_rot_deg": float(np.degrees(rotation.magnitude())),
                "gripper_qpos": float(np.abs(obs["robot0_gripper_qpos"] - stored["robot0_gripper_qpos"]).max()),
                "unforwarded_eef_pos_m": pos_error(unforwarded_obs),
                "step_return_eef_pos_m": pos_error(step_obs),
            }
        record["best_offset"] = min(record["candidates"], key=lambda k: record["candidates"][k]["arm_qpos_max_abs"])
        records.append(record)
    return records


def table(headers, rows):
    """Format rows as a Markdown table.

    Args:
        headers (list): Column names.
        rows (list): Rows of cell values.

    Returns:
        str: Markdown table text.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(path, args, versions, demo, n, settings, zero_step, next_obs_error, reset_errors, frames, replay):
    """Write the Markdown inspection report.

    Args:
        path (Path): Output report path.
        args (argparse.Namespace): Command-line arguments.
        versions (dict): Output of collect_versions.
        demo (dict): Output of load_demo.
        n (int): Number of recorded actions.
        settings (dict): Output of live_settings.
        zero_step (dict): Output of zero_action_step.
        next_obs_error (dict): Max error between next_obs[t] and obs[t+1], per key.
        reset_errors (dict): Output of compare_reset_observations.
        frames (dict): Output of compare_frames.
        replay (list): Output of single_step_replay.

    Returns:
        None: Nothing.
    """
    state_dt = np.diff(demo["states"][:, 0])
    actions = demo["actions"]
    arrays = [("states", demo["states"]), ("actions", actions)] + [(f"obs/{k}", v) for k, v in demo["obs"].items()]
    sections = [
        "# Step 1 inspection: Lift PH low-dim\n\nAll hard checks passed. The measured values below have no pass or fail threshold.",
        f"## Dataset\n\n- Path: `{versions['dataset']['path']}`\n- SHA-256: `{versions['dataset']['sha256']}`\n"
        f"- Demonstrations: {demo['num_demos']}\n- Inspected: `{args.demo}`, N = {n}\n- Attributes: {demo['attrs']}\n"
        f"- `ep_meta` present: {demo['ep_meta'] is not None}\n- Sum of `rewards`: {demo['rewards'].sum():.0f}, sum of `dones`: {demo['dones'].sum()}",
        "## Array shapes\n\n" + table(["Key", "Shape", "dtype"], [(k, v.shape, v.dtype) for k, v in arrays]),
        "## Live settings\n\n" + table(["Setting", "Value"], list(settings.items())),
        f"## Stored `controller_configs`\n\n```json\n{json.dumps(demo['env_args']['env_kwargs']['controller_configs'], indent=2)}\n```",
        "## Timing\n\n"
        f"Stored state time step: min {state_dt.min():.6f} s, max {state_dt.max():.6f} s.",
        "## Action range\n\n" + table(["Channel", "Min", "Max"], [(i, f"{actions[:, i].min():.4f}", f"{actions[:, i].max():.4f}") for i in range(7)]),
        "## Zero-action step\n\n" + table(["Measure", "Value"], [(k, f"{v:.3e}") for k, v in zero_step.items()]),
        "## Observation consistency\n\nMax absolute error over the demonstration.\n\n"
        + table(
            ["Key", "next_obs[t] vs obs[t+1]", "obs[t] vs reset_to, first read", "obs[t] vs reset_to, second read"],
            [(k, f"{next_obs_error[k]:.3e}", f"{reset_errors['first_read'][k]:.3e}", f"{reset_errors['second_read'][k]:.3e}") for k in demo["obs"]],
        ),
        "## `object` components\n\nStored values against a fresh read, in observable order.\n\n"
        + table(["Component", "Dim", "Max error"], [(c["name"], c["dim"], f"{c['max_error']:.3e}") for c in reset_errors["object_components"]]),
        "## Orientation frames\n\n" + table(["Measure", "Value"], list(frames.items())),
        "## Single-step replay\n\nEach row restores `states[t]`, steps with `actions[t+offset]`, and compares with stored step t+1. "
        "EEF rows use a read after `sim.forward()`. The last two columns repeat the EEF position error for a read without `forward` and for the observation that `step` returned.\n\n"
        + table(
            ["t", "Offset", "Arm qpos max abs", "qvel max abs", "EEF pos (m)", "EEF rot (deg)", "Gripper qpos", "Lowest arm error", "No forward (m)", "Step return (m)"],
            [
                (
                    r["t"],
                    o,
                    f"{c['arm_qpos_max_abs']:.3e}",
                    f"{c['qvel_max_abs']:.3e}",
                    f"{c['eef_pos_m']:.3e}",
                    f"{c['eef_rot_deg']:.3e}",
                    f"{c['gripper_qpos']:.3e}",
                    "yes" if o == r["best_offset"] else "",
                    f"{c['unforwarded_eef_pos_m']:.3e}",
                    f"{c['step_return_eef_pos_m']:.3e}",
                )
                for r in replay
                for o, c in r["candidates"].items()
            ],
        ),
    ]
    path.write_text("\n\n".join(sections) + "\n")


def main():
    """Run the checks and write the report and version record.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/robomimic/datasets/lift/ph/low_dim_v15.hdf5"))
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/sim-and-dataset-inspection"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    versions = collect_versions(args.dataset)
    demo = load_demo(args.dataset, args.demo)
    n = check_arrays(demo)

    next_obs_error = {k: float(np.abs(demo["next_obs"][k][:-1] - v[1:]).max()) for k, v in demo["obs"].items()}

    env = make_env(demo)
    settings = live_settings(env)
    require(settings["controller_class"] == "OperationalSpaceController", "arm controller is not OSC")
    require(settings["action_dim"] == 7, "environment action dimension is not 7")
    zero_step = zero_action_step(env)

    restore_model(env, demo)
    reset_errors = compare_reset_observations(env, demo)
    frames = compare_frames(env, demo)
    # Replay error at t = 0 depends on what the simulator ran before, so start from a fresh model load.
    restore_model(env, demo)
    replay = single_step_replay(env, demo, n)

    (args.out_dir / "env_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    write_report(args.out_dir / "report.md", args, versions, demo, n, settings, zero_step, next_obs_error, reset_errors, frames, replay)
    print(f"All hard checks passed. Report: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
