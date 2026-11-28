"""H33: run one checkpoint with Tool Hang phase flags and save every episode for failure analysis.

Run from the repo root:
    uv run python scripts/failure_analysis.py --run checkpoints/h30 --step 1900000
"""

import argparse
import json
from pathlib import Path

import h5py
import jax
import numpy as np
from flax import nnx

from bezier_rfm.datasets import window_loader as wl
from bezier_rfm.eval.rollout import rollout
from bezier_rfm.models.transformer import SplineTransformer
from bezier_rfm.training import load_params
from eval import ROOT, make_policy

PHASES = ("frame grasped", "frame assembled", "tool grasped", "tool hung")


def save_episodes(path, jobs, results, meta):
    """Write all episodes of one run to an npz file.

    Args:
        path (Path): Output file.
        jobs (list): Episode dicts with demo and kind.
        results (list): Rollout results, one per job.
        meta (dict): Run settings, stored as JSON.

    Returns:
        None: Nothing.
    """
    arrays = {"meta": json.dumps(meta), "start": np.array([str(j["demo"]) for j in jobs]),
              "kind": np.array([j["kind"] for j in jobs]), "success": np.array([r["success"] for r in results]),
              "first": np.array([-1 if r["first_success_step"] is None else r["first_success_step"] for r in results])}
    for i, r in enumerate(results):
        for k in ("actions", "achieved_pos", "achieved_rot", "obs", "phases", "call_steps"):
            arrays[f"{i}/{k}"] = r[k]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--step", type=int, required=True, help="checkpoint step")
    ap.add_argument("--n-test", type=int, default=200, help="test env seeds from --test-seed")
    ap.add_argument("--test-seed", type=int, default=100000)
    ap.add_argument("--n-train", type=int, default=6, help="training starts demo_0 ... demo_<n-1>")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--m", type=int, default=1, help="segments executed per policy call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", type=Path, default=ROOT / "low_dim_v15.hdf5", help="demos with the MuJoCo models")
    ap.add_argument("--demos-file", type=Path, default=ROOT / "low_dim_abs_q0start.hdf5", help="env_args source")
    ap.add_argument("--out", type=Path, help="output npz (default: docs/tool_hang/h33/<run>_<step>.npz)")
    args = ap.parse_args()
    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    stats, _ = wl.load_stats(run_dir / "stats.json")
    with h5py.File(args.demos_file, "r") as d:
        env_args = d["data"].attrs["env_args"]
    model = SplineTransformer(nnx.Rngs(params=0, dropout=0), dropout=cfg["dropout"])
    load_params(model, run_dir / f"ema_{args.step:07d}.pkl")
    model.eval()
    policy, _ = make_policy(model, stats, cfg["nfe"], jax.random.PRNGKey(args.seed))
    jobs = [{"demo": args.test_seed + i, "kind": "test"} for i in range(args.n_test)]
    jobs += [{"demo": f"demo_{i}", "kind": "train"} for i in range(args.n_train)]
    results = rollout(env_args, str(args.source), jobs, policy, n_envs=args.n_envs, batch_size=args.n_envs,
                      max_steps=args.max_steps, stop_on_success=True, m=args.m)
    out = args.out or Path("docs/tool_hang/h33") / f"{run_dir.name}_{args.step}.npz"
    save_episodes(out, jobs, results, {**vars(args), "source": str(args.source), "demos_file": str(args.demos_file),
                                       "out": str(out), "phases": PHASES})
    for kind in ("test", "train"):
        ph = [r["phases"].any(axis=0) for j, r in zip(jobs, results) if j["kind"] == kind]
        if ph:
            print(kind, ", ".join(f"{n} {np.mean([p[i] for p in ph]):.2f}" for i, n in enumerate(PHASES)))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
