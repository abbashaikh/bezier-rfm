"""Convert Lift delta actions into absolute grip-site goals in a new HDF5 file."""

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import arm_controller, demo_keys, make_env, require

ACTION_CONVENTION = {
    "type": "absolute",
    "position": "world-frame grip-site target position (m), actions[:, 0:3]",
    "orientation": "rotation vector of the world-frame grip-site target orientation (rad), actions[:, 3:6]",
    "gripper": "original gripper command, actions[:, 6]",
    "goal_source": "OSC_POSE controller goal_pos and goal_ori after set_goal on the stored delta action",
}


def absolute_env_args(env_args):
    """Return a copy of the environment metadata with an absolute-input arm controller.

    Args:
        env_args (dict): Stored delta-controller metadata.

    Returns:
        dict: Metadata that creates an absolute controller.
    """
    result = copy.deepcopy(env_args)
    arm = result["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    require(arm["type"] == "OSC_POSE", "arm controller is not OSC_POSE")
    # robosuite reads input_type. The older control_delta key would only mislead a reader.
    arm.pop("control_delta", None)
    arm["input_type"] = "absolute"
    arm["input_ref_frame"] = "world"
    return result


def convert_demo(env, demo):
    """Convert one demonstration's delta actions to absolute goals.

    Args:
        env (EnvRobosuite): Environment with a delta controller.
        demo (h5py.Group): One stored demonstration.

    Returns:
        np.ndarray: Absolute actions, shape (N, 7).
    """
    states, actions = demo["states"][:], demo["actions"][:]
    env.reset_to({"states": states[0], "model": demo.attrs["model_file"]})
    # Loading a model creates new simulator objects, so fetch the controller afterwards.
    controller = arm_controller(env)
    absolute = np.empty_like(actions)
    for t in range(len(actions)):
        env.reset_to({"states": states[t]})
        # set_goal alone leaves the gripper state and motor commands untouched.
        controller.update(force=True)
        controller.set_goal(actions[t, :6])
        absolute[t, :3] = controller.goal_pos
        absolute[t, 3:6] = Rotation.from_matrix(controller.goal_ori).as_rotvec()
        absolute[t, 6] = actions[t, 6]
    return absolute


def write_output(source, output, converted, env_args, source_sha256):
    """Write the converted dataset, then make it read-only.

    Args:
        source (Path): Original dataset path.
        output (Path): Output dataset path.
        converted (dict): Absolute actions per demonstration name.
        env_args (dict): Original environment metadata.
        source_sha256 (str): Hash of the original dataset.

    Returns:
        None: Nothing.
    """
    temporary = output.with_name(output.name + ".tmp")
    # copyfile does not copy the source's read-only permission.
    shutil.copyfile(source, temporary)
    with h5py.File(temporary, "r+") as f:
        data = f["data"]
        for key, actions in converted.items():
            data[key]["actions"][...] = actions
        data.attrs["env_args"] = json.dumps(absolute_env_args(env_args))
        data.attrs["env_args_delta"] = json.dumps(env_args)
        data.attrs["action_convention"] = json.dumps({**ACTION_CONVENTION, "source_sha256": source_sha256})
    # The rename means a failed run never leaves a partial file at the output path.
    os.replace(temporary, output)
    output.chmod(0o400)


def main():
    """Convert all demonstrations and write the output file.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/lift/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--output", type=Path, default=root / "low_dim_abs.hdf5")
    args = parser.parse_args()
    require(not args.output.exists(), f"{args.output} exists. Delete it first to convert again")

    converted = {}
    with h5py.File(args.source, "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"])
        keys = demo_keys(data)
        env = make_env(env_args, data[f"{keys[0]}/obs"].keys())
        for i, key in enumerate(keys):
            absolute = convert_demo(env, data[key])
            require(absolute.shape == data[key]["actions"].shape, f"{key} has shape {absolute.shape}")
            require(np.isfinite(absolute).all(), f"{key} has non-finite absolute actions")
            converted[key] = absolute
            if i % 50 == 0:
                print(f"Converted {key}", flush=True)
    source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
    write_output(args.source, args.output, converted, env_args, source_sha256)
    print(f"Wrote {len(converted)} demonstrations to {args.output}")


if __name__ == "__main__":
    main()
