"""H33: replay saved episodes of scripts/failure_analysis.py, check that the phase flags repeat, and render videos.

Run from the repo root:
    uv run python scripts/failure_replay.py docs/tool_hang/h33/h30_1900000.npz --starts 100012 100040
"""

import os

# EGL does not work on this machine; glfw does.
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import json
from pathlib import Path

import cv2
import h5py
import imageio
import mujoco
import numpy as np

from bezier_rfm.datasets.robomimic import arm_controller, load_demo_model, make_env, read_result
from bezier_rfm.datasets.windows import OBS_KEYS
from failure_plots import STAGES

SIZE = 480


def replay(env, data, start, actions, cameras, write):
    """Reset one start as the rollout does, run the saved commands, record phase flags, and pass each frame on.

    Args:
        env (EnvRobosuite): Environment with the absolute OSC controller.
        data (h5py.Group): Demo data group for training starts.
        start (str): Env seed (digits) or demo name.
        actions (np.ndarray): Executed commands, shape (S, 7).
        cameras (tuple): Camera names, one panel each.
        write (callable): write(t, frame) for t = 0..S, frame shape (SIZE, SIZE * len(cameras), 3).

    Returns:
        np.ndarray: Phase flags, shape (S, 4).
    """
    base = env.env
    if start.isdigit():
        # Same start as env_worker in bezier_rfm/eval/rollout.py.
        np.random.seed(int(start))
        env.reset()
    else:
        load_demo_model(env, data[start])
    arm_controller(env).new_update = True
    renderer = mujoco.Renderer(base.sim.model._model, SIZE, SIZE)

    def render():
        # No forward call here: one more mj_forward per step than the rollout changes the later physics.
        panels = []
        for cam in cameras:
            renderer.update_scene(base.sim.data._data, camera=cam)
            panels.append(renderer.render().copy())
        return np.concatenate(panels, axis=1)

    gripper = base.robots[0].gripper
    # Same forward call after the reset as env_worker (read_result).
    read_result(env)
    # Frames go straight to the writer: one episode of frames does not fit in memory next to a rollout.
    write(0, render())
    flags = []
    for t, a in enumerate(actions, 1):
        env.step(a)
        # Same calls and order as the "step" branch of env_worker, so the flags match the saved ones.
        base.sim.forward()
        read_result(env)
        flags.append([base._check_grasp(gripper, base.frame), base._check_frame_assembled(),
                      base._check_grasp(gripper, base.tool), base._check_tool_on_frame()])
        write(t, render())
    renderer.close()
    return np.array(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, help="episodes from scripts/failure_analysis.py")
    ap.add_argument("--starts", nargs="+", required=True, help="episode starts to replay (env seed or demo name)")
    ap.add_argument("--cameras", nargs="+", default=["frontview", "agentview"])
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    d = np.load(args.npz)
    meta = json.loads(str(d["meta"]))
    with h5py.File(meta["demos_file"], "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    env = make_env(env_args, OBS_KEYS)
    data = h5py.File(meta["source"], "r")["data"]
    out_dir = args.npz.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    starts = [str(s) for s in d["start"]]
    for s in args.starts:
        i = starts.index(s)
        saved = d[f"{i}/phases"]
        stage = np.cumprod(np.logical_or.accumulate(saved, axis=0), axis=1).sum(axis=1)
        out = out_dir / f"replay_{s}.mp4"
        with imageio.get_writer(out, fps=args.fps) as w:

            def write(t, fr):
                label = f"{s}  step {t}  {STAGES[stage[max(t - 1, 0)]]}"
                cv2.putText(fr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                w.append_data(fr)

            flags = replay(env, data, s, d[f"{i}/actions"], tuple(args.cameras), write)
        match = flags.shape == saved.shape and bool((flags == saved).all())
        print(f"{s}: success {bool(d['success'][i])}, phase flags repeat {match}, saved {out}")


if __name__ == "__main__":
    main()
