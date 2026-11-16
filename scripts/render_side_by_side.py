"""Replay the q0-start reference and Bézier candidates of one Lift or Tool Hang demonstration and render them side by side."""

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

from bezier_rfm.datasets.robomimic import arm_controller, load_demo_model, make_env, require

SIZE = 480


def replay_images(env, demo, actions, camera="frontview", hold=0):
    """Restore s_0 with a refreshed controller, run the actions, and render one camera after each step.

    Args:
        env (EnvRobosuite): Environment with an absolute OSC_POSE controller.
        demo (h5py.Group): Demonstration that supplies the model and s_0.
        actions (np.ndarray): Absolute actions, shape (N, 7).
        camera (str): Camera name.
        hold (int): Largest number of extra steps that repeat the last action until success.

    Returns:
        tuple: Images, N + 1 + (hold steps used) of shape (SIZE, SIZE, 3), and the success flag.
    """
    load_demo_model(env, demo)
    # Same start as evaluate_commands(start="refreshed").
    arm_controller(env).new_update = True
    renderer = mujoco.Renderer(env.env.sim.model._model, SIZE, SIZE)

    def render():
        # Same forward call per step as read_state and screen_replay, so the replay matches their nominal runs.
        env.env.sim.forward()
        renderer.update_scene(env.env.sim.data._data, camera=camera)
        return renderer.render().copy()

    images = [render()]
    success = False
    for action in actions:
        env.step(action)
        images.append(render())
        # Tool Hang counts success at any step; render() already ran the forward pass.
        success = success or bool(env.env._check_success())
    for _ in range(hold):
        if success:
            break
        env.step(actions[-1])
        images.append(render())
        success = bool(env.env._check_success())
    if hold == 0:
        success = bool(env.env._check_success())
    renderer.close()
    return images, success


def main():
    """Render one mp4 with the reference and every candidate as panels.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["lift", "tool_hang"], default="lift")
    task = parser.parse_known_args()[0].task
    root = Path(f"data/robomimic/datasets/{task}/ph")
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--reference", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--candidates", type=Path, nargs="+", default=[root / "bezier_k5_2.hdf5"])
    parser.add_argument("--out-dir", type=Path, default=Path(f"docs/{task}/bezier-fit/videos"))
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--hold", type=int, default=0 if task == "lift" else 60, help="Extra steps with the last action until success")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    with h5py.File(args.source, "r") as source_file, h5py.File(args.reference, "r") as ref_file:
        demo = source_file[f"data/{args.demo}"]
        env = make_env(json.loads(ref_file["data"].attrs["env_args"]), demo["obs"].keys())
        require(arm_controller(env).input_type == "absolute", "environment is not absolute")
        panels.append(("Reference (q0-start replay)", *replay_images(env, demo, ref_file[f"data/{args.demo}/actions"][:], args.camera, args.hold)))
        for path in args.candidates:
            with h5py.File(path, "r") as cand_file:
                meta = json.loads(cand_file["data"].attrs["bezier"])
                actions = cand_file[f"data/{args.demo}/actions"][:]
            label = f"Bezier {tuple(meta['ks'])}" if meta.get("ks") else f"Bezier {meta['variant']}, {meta['density']:g}/s"
            panels.append((label, *replay_images(env, demo, actions, args.camera, args.hold)))

    word = "lifted" if task == "lift" else "success"
    frames = []
    # Panels can differ in length by their hold steps; a finished panel keeps its last frame.
    for t in range(max(len(images) for _, images, _ in panels)):
        row = []
        for label, images, success in panels:
            image = images[min(t, len(images) - 1)].copy()
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            footer = f"t = {min(t, len(images) - 1)}" + (f"  {word}: {success}" if t >= len(images) - 1 else "")
            cv2.putText(image, footer, (10, SIZE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            row.append(image)
        frames.append(np.concatenate(row, axis=1))
    # Hold the last frame for 1 s so that the outcome can be read.
    frames += [frames[-1]] * 20
    out = args.out_dir / f"{args.demo}_side_by_side.mp4"
    imageio.mimwrite(out, frames, fps=20, macro_block_size=1)
    print(f"Wrote {out}: " + ", ".join(f"{label} {word} {success}" for label, _, success in panels))


if __name__ == "__main__":
    main()
