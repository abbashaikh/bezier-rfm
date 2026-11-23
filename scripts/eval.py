"""H30: evaluate saved EMA checkpoints of a training run with closed-loop Tool Hang rollouts (plan.md 6.2, 9.3).

Run from the repo root, next to scripts/train.py (same --run):
    uv run python scripts/eval.py --run checkpoints/h30 --watch          # evaluate each new checkpoint until the last
    uv run python scripts/eval.py --run checkpoints/h30 --steps 19000    # one checkpoint
The W&B key comes only from the WANDB_API_KEY environment variable.
"""

import argparse
import json
import re
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bezier_rfm.datasets import window_loader as wl
from bezier_rfm.eval.rollout import rollout
from bezier_rfm.flow_matching import spline
from bezier_rfm.models.transformer import SplineTransformer
from bezier_rfm.training import load_params

ROOT = Path("data/robomimic/datasets/tool_hang/ph")


def make_policy(model, stats, nfe, key):
    """Build the rollout policy: one sampled window per request, plus a log of inference times.

    Args:
        model (nnx.Module): EMA model in eval mode.
        stats (spline.Stats): Scales from stats.json.
        nfe (int): Sampler steps.
        key (jax.Array): PRNG key of the prior draws.

    Returns:
        tuple: policy(requests) -> list of window dicts, and the list of call times in s.
    """
    @jax.jit
    def run(model, obs, q0_pos, q0_rot, v0, key):
        cond = spline.make_cond_from(obs, q0_pos, q0_rot, v0, stats)
        return spline.sample(model, spline.sample_prior(key, q0_pos, q0_rot, stats), cond, stats, nfe)

    times, calls = [], [0]

    def policy(requests):
        f = lambda k: jnp.asarray(np.stack([r[k] for r in requests]), jnp.float32)
        t0 = time.perf_counter()
        x = run(model, f("obs"), f("q0_pos"), f("q0_rot"), f("v0"), jax.random.fold_in(key, calls[0]))
        x = jax.device_get(x)
        times.append(time.perf_counter() - t0)
        calls[0] += 1
        k = spline.K
        out = []
        for i, r in enumerate(requests):
            # Index 0 of the poses is q_0 / H_0, which the executor replaces.
            out.append({"q_pos": np.concatenate([r["q0_pos"][None], x.pos[i, :k]]).astype(float),
                        "q_rot": np.concatenate([r["q0_rot"][None], x.rot[i, :k]]).astype(float),
                        "H_pos": np.concatenate([r["q0_pos"][None], x.pos[i, k:]]).astype(float),
                        "H_rot": np.concatenate([r["q0_rot"][None], x.rot[i, k:]]).astype(float),
                        "z": float(x.z[i]), "logits": x.ell[i].astype(float), "gripper": x.g[i].astype(float)})
        return out

    return policy, times


def tracking_gap(res):
    """Compute the gap between commanded and achieved grip-site poses after each step of one episode.

    Args:
        res (dict): Rollout result with actions (S, 7), achieved_pos (S, 3), achieved_rot (S, 3, 3).

    Returns:
        tuple: Position gap in mm and rotation gap in deg, each shape (S,).
    """
    from scipy.spatial.transform import Rotation

    a = res["actions"]
    n = len(a)
    pos = np.linalg.norm(a[:, :3] - res["achieved_pos"][:n], axis=1) * 1e3
    rel = Rotation.from_rotvec(a[:, 3:6]).inv() * Rotation.from_matrix(res["achieved_rot"][:n])
    return pos, np.degrees(rel.magnitude())


def evaluate(path, model, stats, cfg, args, env_args):
    """Run all test and training-start episodes for one checkpoint.

    Args:
        path (Path): EMA checkpoint.
        model (nnx.Module): Model with the run's structure.
        stats (spline.Stats): Scales.
        cfg (dict): Run config.
        args (Namespace): Command-line arguments.
        env_args (str): Environment metadata (absolute OSC controller).

    Returns:
        dict: Scalar results of the checkpoint.
    """
    load_params(model, path)
    model.eval()
    policy, times = make_policy(model, stats, args.nfe or cfg["nfe"], jax.random.PRNGKey(args.seed))
    jobs = [{"demo": args.test_seed + i, "kind": "test"} for i in range(args.n_test)]
    jobs += [{"demo": f"demo_{i}", "kind": "train"} for i in range(args.n_train)]
    t0 = time.time()
    results = rollout(env_args, str(args.source), jobs, policy, n_envs=args.n_envs, batch_size=args.n_envs,
                      max_steps=args.max_steps, stop_on_success=True)
    gaps = [tracking_gap(r) for r in results]
    test = [r["success"] for j, r in zip(jobs, results) if j["kind"] == "test"]
    train = [r["success"] for j, r in zip(jobs, results) if j["kind"] == "train"]
    # The first call of a process includes compilation.
    call_ms = 1e3 * np.array(times[1:] if len(times) > 1 else times)
    return {
        "rollout/success_test": float(np.mean(test)) if test else float("nan"),
        "rollout/success_train": float(np.mean(train)) if train else float("nan"),
        "rollout/inference_ms_median": float(np.median(call_ms)),
        "rollout/inference_ms_max": float(call_ms.max()),
        "rollout/gap_pos_mm_mean": float(np.mean([g[0].mean() for g in gaps])),
        "rollout/gap_pos_mm_max": float(max(g[0].max() for g in gaps)),
        "rollout/gap_rot_deg_mean": float(np.mean([g[1].mean() for g in gaps])),
        "rollout/gap_rot_deg_max": float(max(g[1].max() for g in gaps)),
        "rollout/steps_mean": float(np.mean([r["steps"] for r in results])),
        "rollout/calls_mean": float(np.mean([r["calls"] for r in results])),
        "rollout/wall_s": time.time() - t0,
        "episodes": [{"start": j["demo"], "kind": j["kind"], "success": r["success"], "first": r["first_success_step"],
                      "steps": r["steps"], "gap_pos_mm_max": float(g[0].max()), "gap_rot_deg_max": float(g[1].max())}
                     for j, r, g in zip(jobs, results, gaps)],
    }


def init_wandb(cfg, run_dir, mode):
    """Attach to the W&B run of the training run as a second writer.

    Args:
        cfg (dict): Run config with wandb_id and wandb_project.
        run_dir (Path): Run directory.
        mode (str): online, offline or disabled.

    Returns:
        wandb.Run: The run, or None when disabled.
    """
    if mode == "disabled":
        return None
    import wandb

    settings = wandb.Settings(mode="shared", x_primary=False, x_label="eval", x_update_finish_state=False) \
        if mode == "online" else wandb.Settings(mode=mode)
    run = wandb.init(project=cfg["wandb_project"], id=cfg["wandb_id"], resume="allow", dir=str(run_dir / "eval"),
                     settings=settings)
    run.define_metric("rollout/train_step")
    run.define_metric("rollout/*", step_metric="rollout/train_step")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--steps", type=int, nargs="*", help="checkpoint steps to evaluate (default: all saved)")
    ap.add_argument("--watch", action="store_true", help="wait for new checkpoints until the last one of the run")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks in --watch mode")
    ap.add_argument("--n-test", type=int, default=50, help="test env seeds from --test-seed")
    ap.add_argument("--test-seed", type=int, default=100000)
    ap.add_argument("--n-train", type=int, default=6, help="training starts demo_0 ... demo_<n-1>")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--nfe", type=int, help="sampler steps (default: the run config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", type=Path, default=ROOT / "low_dim_v15.hdf5", help="demos with the MuJoCo models")
    ap.add_argument("--demos-file", type=Path, default=ROOT / "low_dim_abs_q0start.hdf5", help="env_args source")
    ap.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    args = ap.parse_args()
    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    stats, _ = wl.load_stats(run_dir / "stats.json")
    with h5py.File(args.demos_file, "r") as d:
        env_args = d["data"].attrs["env_args"]
    model = SplineTransformer(nnx.Rngs(params=0, dropout=0), dropout=cfg["dropout"])
    out_file = run_dir / "eval.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(line)["step"] for line in out_file.read_text().splitlines()}
    wb = init_wandb(cfg, run_dir, args.wandb)
    last = cfg["steps"]

    while True:
        saved = {int(re.findall(r"\d+", p.name)[0]): p for p in run_dir.glob("ema_*.pkl")}
        todo = sorted(s for s in saved if s not in done and (not args.steps or s in args.steps))
        for step in todo:
            r = evaluate(saved[step], model, stats, cfg, args, env_args)
            r["step"] = step
            with open(out_file, "a") as f:
                f.write(json.dumps(r) + "\n")
            done.add(step)
            print(f"step {step}: success test {r['rollout/success_test']:.2f} train {r['rollout/success_train']:.2f} "
                  f"inference {r['rollout/inference_ms_median']:.1f} ms gap {r['rollout/gap_pos_mm_mean']:.2f} mm "
                  f"{r['rollout/gap_rot_deg_mean']:.2f} deg ({r['rollout/wall_s']:.0f} s)", flush=True)
            if wb is not None:
                wb.log({"rollout/train_step": step, **{k: v for k, v in r.items() if k.startswith("rollout/")}})
        if not args.watch or last in done:
            break
        time.sleep(args.poll)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
