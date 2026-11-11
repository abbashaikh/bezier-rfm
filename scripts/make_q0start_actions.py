"""Copy the absolute-action Lift dataset with action 0 set to the start grip-site pose."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import demo_keys, load_demo_model, make_env, read_result, require

T0_RULE = "actions[0, :6] is the grip-site pose at state 0 (from the float64 simulator state); actions[0, 6] and all later actions are unchanged"


def start_actions(env, data, keys):
    """Return the first action of every demonstration with its pose part set to the start pose.

    Args:
        env (EnvRobosuite): Environment that can load the demonstrations.
        data (h5py.Group): The "data" group of the absolute-action dataset.
        keys (list): Demonstration names.

    Returns:
        dict: Demonstration name -> new first action, shape (7,).
    """
    first = {}
    for key in keys:
        load_demo_model(env, data[key])
        pose = read_result(env)
        action = data[key]["actions"][0].copy()
        action[:3] = pose["pos"]
        action[3:6] = Rotation.from_matrix(pose["rot"]).as_rotvec()
        require(np.isfinite(action).all(), f"{key} has a non-finite first action")
        first[key] = action
    return first


def main():
    """Write the q0-start copy of the absolute-action dataset.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/lift/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute", type=Path, default=root / "low_dim_abs.hdf5")
    parser.add_argument("--output", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    args = parser.parse_args()
    require(not args.output.exists(), f"{args.output} exists. Delete it first to write it again")

    with h5py.File(args.absolute, "r") as f:
        data = f["data"]
        keys = demo_keys(data)
        env = make_env(json.loads(data.attrs["env_args"]), data[f"{keys[0]}/obs"].keys())
        first = start_actions(env, data, keys)
        convention = json.loads(data.attrs["action_convention"])

    temporary = args.output.with_name(args.output.name + ".tmp")
    # copyfile does not copy the source's read-only permission.
    shutil.copyfile(args.absolute, temporary)
    with h5py.File(temporary, "r+") as f:
        data = f["data"]
        for key, action in first.items():
            data[key]["actions"][0] = action
        data.attrs["action_convention"] = json.dumps({**convention, "t0": T0_RULE, "absolute_sha256": hashlib.sha256(args.absolute.read_bytes()).hexdigest()})
    # The rename means a failed run never leaves a partial file at the output path.
    os.replace(temporary, args.output)
    args.output.chmod(0o400)
    print(f"Wrote {len(first)} demonstrations to {args.output}")


if __name__ == "__main__":
    main()
