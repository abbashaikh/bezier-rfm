"""Build diagnostic traces of recorded Lift demonstrations and their absolute-action replays."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from bezier_rfm.datasets.robomimic import arm_controller, demo_keys, make_env, require
from bezier_rfm.eval.lift_diagnostics import evaluate_commands, record_trace, write_trace

ALIGNMENT = "Action t runs from state t to state t + 1, over [t_start, t_end] = [0.05 t, 0.05 (t + 1)] s."
T0_NOTE = (
    "actions[0] is the goal from the actual start pose. The recording was driven by the stale goal "
    "(nominal start pose + first delta), so the recorded trace's tracking error at t = 0 is not meaningful."
)
TERMINATION = "ignore_done is True and the horizon is 1000 steps, so episodes never terminate. Only success is recorded."


def sha256(path):
    """Return the SHA-256 of a file.

    Args:
        path (Path): File path.

    Returns:
        str: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_attrs(args, env, abs_data):
    """Collect metadata shared by every trace.

    Args:
        args (argparse.Namespace): Command-line arguments.
        env (EnvRobosuite): Absolute-controller environment.
        abs_data (h5py.Group): Converted "data" group.

    Returns:
        dict: Metadata.
    """
    base = env.env
    controller = arm_controller(env)
    return {
        "source_file": str(args.source),
        "source_sha256": sha256(args.source),
        "absolute_file": str(args.absolute),
        "absolute_sha256": sha256(args.absolute),
        "env_args": json.loads(abs_data.attrs["env_args"]),
        "action_convention": json.loads(abs_data.attrs["action_convention"]),
        "control_freq": float(base.control_freq),
        "model_timestep": float(base.model_timestep),
        "substeps": int(round(base.control_timestep / base.model_timestep)),
        "controller": {"input_type": controller.input_type, "input_ref_frame": controller.input_ref_frame, "ref_name": controller.ref_name},
        "versions": json.loads(args.versions.read_text()),
        "alignment": ALIGNMENT,
        "t0_note": T0_NOTE,
        "termination": TERMINATION,
    }


def main():
    """Build traces and write them to one HDF5 file.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/lift/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--absolute", type=Path, default=root / "low_dim_abs.hdf5")
    parser.add_argument("--versions", type=Path, default=Path("docs/sim-and-dataset-inspection/env_versions.json"))
    parser.add_argument("--demos", nargs="+", default=["demo_0"], help='Demonstration names, or "all"')
    parser.add_argument("--start", default="refreshed", choices=["refreshed", "stale"])
    parser.add_argument("--out", type=Path, default=Path("docs/diagnostics/baseline_traces.hdf5"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source, "r") as source_file, h5py.File(args.absolute, "r") as abs_file:
        source, output = source_file["data"], abs_file["data"]
        names = demo_keys(source) if args.demos == ["all"] else args.demos
        env = make_env(json.loads(output.attrs["env_args"]), source[f"{names[0]}/obs"].keys())
        require(arm_controller(env).input_type == "absolute", "environment is not absolute")
        attrs = common_attrs(args, env, output)
        tmp = args.out.with_suffix(".tmp")
        with h5py.File(tmp, "w") as out:
            write_trace(out.create_group("meta"), {}, attrs)
            for i, name in enumerate(names):
                actions = output[f"{name}/actions"][:]
                recorded = record_trace(env, source[name], actions)
                write_trace(out.create_group(f"{name}/recorded"), recorded, {"demo": name, "trajectory": "recorded", "start": "recorded", "command_source": "low_dim_abs.hdf5 actions"})
                replay = evaluate_commands(env, source[name], actions, start=args.start)
                write_trace(out.create_group(f"{name}/replay_absolute"), replay, {"demo": name, "trajectory": "replay_absolute", "start": args.start, "command_source": "low_dim_abs.hdf5 actions"})
                if i % 25 == 0:
                    print(f"Finished {name}", flush=True)
        tmp.replace(args.out)
    print(f"Traces: {args.out}")


if __name__ == "__main__":
    main()
