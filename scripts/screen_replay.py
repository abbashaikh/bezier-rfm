"""Screen demonstrations by q0-start replay success, with hold steps and small command noise, in parallel."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import arm_controller, demo_keys, load_demo_model, make_env

WORKER = {}


def init_worker(source, actions):
    """Open the datasets and create one environment per worker process.

    Args:
        source (str): Original dataset path.
        actions (str): Absolute-action dataset path.

    Returns:
        None: Nothing.
    """
    WORKER["source"] = h5py.File(source, "r")["data"]
    WORKER["actions"] = h5py.File(actions, "r")["data"]
    keys = demo_keys(WORKER["source"])
    WORKER["env"] = make_env(json.loads(WORKER["actions"].attrs["env_args"]), WORKER["source"][f"{keys[0]}/obs"].keys())


def perturb(actions, seed, pos_sigma, rot_sigma):
    """Add Gaussian noise to the pose commands after action 0.

    Args:
        actions (np.ndarray): Absolute actions, shape (N, 7).
        seed (int): Random seed; 0 returns the actions unchanged.
        pos_sigma (float): Position noise per axis (m).
        rot_sigma (float): Rotation noise per axis (rad).

    Returns:
        np.ndarray: Perturbed actions, shape (N, 7).
    """
    if seed == 0:
        return actions
    rng = np.random.default_rng(seed)
    out = actions.copy()
    n = len(actions) - 1
    out[1:, :3] += pos_sigma * rng.normal(size=(n, 3))
    noise = Rotation.from_rotvec(rot_sigma * rng.normal(size=(n, 3)))
    out[1:, 3:6] = (noise * Rotation.from_rotvec(actions[1:, 3:6])).as_rotvec()
    return out


def success_now(base):
    """Return whether the task succeeds in the up-to-date simulator state.

    Args:
        base (ToolHang): The robosuite environment.

    Returns:
        bool: Success flag.
    """
    # After mj_step, site poses describe the qpos before the last integration.
    base.sim.forward()
    return bool(base._check_success())


def run(job):
    """Replay one demonstration once, then hold the last command.

    Args:
        job (tuple): Demonstration name, seed, position sigma (m), rotation sigma (rad), hold steps, save path flag.

    Returns:
        dict: Success flags, first success step, the two Tool Hang checks at the end of the actions, and the
            grip-site path after each action (nominal run only, when requested).
    """
    name, seed, pos_sigma, rot_sigma, hold, save_path = job
    env, base = WORKER["env"], WORKER["env"].env
    actions = perturb(WORKER["actions"][f"{name}/actions"][:], seed, pos_sigma, rot_sigma)
    load_demo_model(env, WORKER["source"][name])
    # Refreshed start, as replay_q0start.
    arm_controller(env).new_update = True
    first = None
    site = base.robots[0].eef_site_id["right"]
    path_pos, path_rot = [], []
    for t, action in enumerate(actions):
        env.step(action)
        if first is None and success_now(base):
            first = t + 1
        if save_path and seed == 0:
            # success_now ran sim.forward, so the site pose is up to date.
            path_pos.append(base.sim.data.site_xpos[site].copy())
            path_rot.append(base.sim.data.site_xmat[site].reshape(3, 3).copy())
    base.sim.forward()
    end = {"frame_assembled": bool(base._check_frame_assembled()), "tool_on_frame": bool(base._check_tool_on_frame())}
    first_hold = None
    for h in range(hold):
        env.step(actions[-1])
        if first is None and first_hold is None and success_now(base):
            first_hold = h + 1
    out = {
        "demo": name,
        "seed": seed,
        "n": len(actions),
        "success_in_actions": first is not None,
        "first_success_state": first,
        "success_with_hold": first is not None or first_hold is not None,
        "hold_steps_needed": 0 if first is not None else first_hold,
        **end,
    }
    if path_pos:
        out["path"] = (np.array(path_pos), np.array(path_rot))
    return out


def main():
    """Run all jobs in a process pool and write one JSON file.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/tool_hang/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--actions", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--demos", nargs="+", default=["all"], help='Demonstration names, or "all"')
    parser.add_argument("--seeds", type=int, default=5, help="Noisy runs per demo, in addition to the nominal run")
    parser.add_argument("--pos-sigma", type=float, default=2e-4)
    parser.add_argument("--rot-sigma", type=float, default=2e-4)
    parser.add_argument("--hold", type=int, default=60)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--save-paths", action="store_true", help="Store the grip-site path of each nominal run next to the JSON file")
    parser.add_argument("--out", type=Path, default=Path("docs/tool_hang/screening/screen_q0start.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source, "r") as f:
        names = demo_keys(f["data"]) if args.demos == ["all"] else args.demos
    jobs = [(name, seed, args.pos_sigma, args.rot_sigma, args.hold, args.save_paths) for name in names for seed in range(args.seeds + 1)]
    results = []
    with Pool(args.workers, initializer=init_worker, initargs=(str(args.source), str(args.actions))) as pool:
        for i, result in enumerate(pool.imap_unordered(run, jobs)):
            results.append(result)
            if i % 50 == 0:
                print(f"Finished {i + 1} of {len(jobs)} runs", flush=True)
    results.sort(key=lambda r: (int(r["demo"].split("_")[1]), r["seed"]))
    paths = {}
    for r in results:
        if "path" in r:
            pos, rot = r.pop("path")
            paths[f"{r['demo']}_pos"], paths[f"{r['demo']}_rot"] = pos, rot
    if paths:
        np.savez_compressed(args.out.with_suffix(".paths.npz"), **paths)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.out.write_text(json.dumps({"settings": settings, "runs": results}, indent=1))
    print(f"Wrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
