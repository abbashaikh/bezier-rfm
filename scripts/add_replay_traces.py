"""Replay an absolute- or delta-action Lift or Tool Hang dataset and add the traces as a new group in an existing traces file."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from bezier_rfm.datasets.robomimic import arm_controller, demo_keys, make_env, require
from bezier_rfm.eval.lift_diagnostics import evaluate_commands, write_trace

# The baseline groups come from build_diagnostic_traces.py and must stay as built.
PROTECTED = ("recorded", "replay_absolute")


def main():
    """Replay the actions and write one trace group per demonstration.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["lift", "tool_hang"], default="lift")
    task = parser.parse_known_args()[0].task
    root = Path(f"data/robomimic/datasets/{task}/ph")
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--actions", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--trajectory", default="replay_q0start", help="Group name under each demonstration")
    parser.add_argument("--demos", nargs="+", default=["demo_0"], help='Demonstration names, or "all"')
    parser.add_argument("--start", default="refreshed", choices=["refreshed", "stale"])
    parser.add_argument("--traces", type=Path, default=Path("docs/diagnostics/baseline_traces.hdf5" if task == "lift" else f"docs/{task}/diagnostics/traces.hdf5"))
    parser.add_argument("--controller", default="absolute", choices=["absolute", "delta"], help="delta replays the source file's own actions")
    args = parser.parse_args()
    require(args.trajectory not in PROTECTED, f"{args.trajectory} is a baseline group")
    if args.controller == "delta":
        args.actions = args.source
    actions_sha256 = hashlib.sha256(args.actions.read_bytes()).hexdigest()

    with h5py.File(args.source, "r") as source_file, h5py.File(args.actions, "r") as actions_file, h5py.File(args.traces, "r+") as out:
        source, data = source_file["data"], actions_file["data"]
        names = demo_keys(source) if args.demos == ["all"] else args.demos
        env = make_env(json.loads(data.attrs["env_args"]), source[f"{names[0]}/obs"].keys())
        require(arm_controller(env).input_type == args.controller, f"environment is not {args.controller}")
        for i, name in enumerate(names):
            require(name in out, f"{name} has no baseline traces")
            trace = evaluate_commands(env, source[name], data[f"{name}/actions"][:], start=args.start)
            # A rerun replaces this trajectory only.
            if args.trajectory in out[name]:
                del out[name][args.trajectory]
            attrs = {"demo": name, "trajectory": args.trajectory, "start": args.start, "command_source": f"{args.actions.name} actions", "actions_sha256": actions_sha256, "controller": args.controller}
            # A delta trace's cmd_* fields hold the raw delta input, not a pose; goal_* hold the controller goals.
            write_trace(out.create_group(f"{name}/{args.trajectory}"), trace, attrs)
            if i % 25 == 0:
                print(f"Finished {name}", flush=True)
    print(f"Added {args.trajectory} for {len(names)} demonstrations to {args.traces}")


if __name__ == "__main__":
    main()
