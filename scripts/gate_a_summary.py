"""Stage 0 Gate A: summarize the last checkpoints of a run per execution horizon m.

    uv run python scripts/gate_a_summary.py --run checkpoints/h30
"""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(records, last):
    """Compute the Gate A numbers over the last checkpoints of one result file.

    Args:
        records (list): Records of scripts/eval.py, one per checkpoint.
        last (int): Number of latest checkpoints to use.

    Returns:
        dict: Checkpoint steps and the Gate A numbers.
    """
    recs = sorted(records, key=lambda r: r["step"])[-last:]
    test = [r["rollout/success_test"] for r in recs]
    steps = np.mean([r["rollout/steps_mean"] for r in recs])
    calls = np.mean([r["rollout/calls_mean"] for r in recs])
    first = [e["first"] for r in recs for e in r["episodes"] if e["kind"] == "test" and e["first"] is not None]
    return {"steps": [r["step"] for r in recs], "last_mean": np.mean(test), "best": max(test), "calls": calls,
            "steps_per_call": steps / calls, "first": np.mean(first) if first else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--last", type=int, default=10, help="number of latest checkpoints")
    args = ap.parse_args()
    for path in sorted(Path(args.run).glob("eval*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        m = records[0].get("m", 1)
        s = summarize(records, args.last)
        print(f"{path.name} (m = {m}): {len(s['steps'])} checkpoints {s['steps'][0]} ... {s['steps'][-1]}")
        if len(s["steps"]) < args.last:
            print(f"  warning: fewer than {args.last} checkpoints")
        print(f"  Last-{args.last} success (test) {s['last_mean']:.3f}, best {s['best']:.2f}")
        print(f"  calls per episode {s['calls']:.1f}, steps per call {s['steps_per_call']:.2f}, "
              f"mean first-success step (test) {s['first']:.1f}")


if __name__ == "__main__":
    main()
