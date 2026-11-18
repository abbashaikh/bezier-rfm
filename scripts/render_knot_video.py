"""Render one Tool Hang demo as raw, S-fit and (optional) E-fit replays side by side, with a time stamp and a knot-density strip (H26, H27)."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from render_side_by_side import SIZE, replay_images

from bezier_rfm.datasets.robomimic import arm_controller, make_env, require
from bezier_rfm.fitting.density import speed_weight
from bezier_rfm.fitting.knots import CONTROL_DT, clock_h, dense_intervals, error_knots, euclidean_proxy, step_density

STRIP_H = 260


def density_strip(d_s, d_e, intervals, width):
    """Draw the S and E density with the marked intervals, and return the image and the time-to-pixel map.

    Args:
        d_s (np.ndarray): S density per step, shape (N,).
        d_e (np.ndarray | None): E density per step, shape (N,), or None for S only.
        intervals (list): (start, end) step indices of the marked intervals.
        width (int): Image width, px.

    Returns:
        tuple: RGB image (STRIP_H, width, 3) and a function from time (s) to x pixel.
    """
    t = CONTROL_DT * np.arange(len(d_s))
    fig = plt.figure(figsize=(width / 100, STRIP_H / 100), dpi=100)
    ax = fig.add_axes([0.04, 0.2, 0.94, 0.55])
    for i, (a, b) in enumerate(intervals, 1):
        ax.axvspan(t[a], t[b - 1] + CONTROL_DT, color="gold", alpha=0.4, lw=0)
        ax.text(0.5 * (t[a] + t[b - 1]), 1.02, f"#{i}", transform=ax.get_xaxis_transform(), ha="center", fontsize=11, weight="bold")
    ax.step(t, d_s, where="post", color="tab:blue", lw=1.5, label="S (segment clock)")
    if d_e is not None:
        ax.step(t, d_e, where="post", color="tab:red", lw=1.5, label="E (error-driven)")
    ax.set(xlim=(0, t[-1] + 3), ylabel="segments/s")
    ax.set_xlabel("t (s)" + ("; yellow: E density >= 2 x S density for >= 0.5 s" if d_e is not None else ""), fontsize=9)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0, x1 = ax.transData.transform([(0, 0), (t[-1] + 3, 0)])[:, 0]
    plt.close(fig)
    return image, lambda time: int(round(x0 + (x1 - x0) * time / (t[-1] + 3)))


def main():
    """Replay the raw, S-fit and (optional) E-fit action sequences of one demo and write the mp4.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/tool_hang/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default="demo_16")
    parser.add_argument("--source", type=Path, default=root / "low_dim_v15.hdf5")
    parser.add_argument("--reference", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--fit-s", type=Path, default=root / "bezier_h26_S.hdf5")
    parser.add_argument("--fit-e", type=Path, default=root / "bezier_h26_E.hdf5")
    parser.add_argument("--s-only", action="store_true", help="Leave out the E fit")
    parser.add_argument("--tag", default="h26", help="Output file is <demo>_<tag>.mp4")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/tool_hang/h26/videos"))
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--hold", type=int, default=60)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.reference, "r") as ref, h5py.File(args.fit_s, "r") as fs:
        raw = ref[f"data/{args.demo}/actions"][:]
        rule = json.loads(fs["data"].attrs["bezier"])
        h_s = fs[f"data/{args.demo}/bezier"].attrs["h"]
        # The stored knots must be the ones the rules give for the raw commands.
        require(np.allclose(h_s, clock_h(speed_weight(raw), rule["c"], rule["h_min"])), "S knots differ from the rule")
        sequences = [("Raw commands (q0-start)", raw),
                     (f"S fit, c = {rule['c']:.2f}, K = {len(h_s)}", fs[f"data/{args.demo}/actions"][:])]
        h_e = None
        if not args.s_only:
            with h5py.File(args.fit_e, "r") as fe:
                h_e = fe[f"data/{args.demo}/bezier"].attrs["h"]
                require(np.allclose(h_e, error_knots(euclidean_proxy(raw), len(h_s), rule["h_min"])), "E knots differ from the rule")
                sequences.append((f"E fit, K = {len(h_e)}", fe[f"data/{args.demo}/actions"][:]))
        with h5py.File(args.source, "r") as source:
            demo = source[f"data/{args.demo}"]
            env = make_env(json.loads(ref["data"].attrs["env_args"]), demo["obs"].keys())
            require(arm_controller(env).input_type == "absolute", "environment is not absolute")
            panels = [(label, *replay_images(env, demo, actions, args.camera, args.hold)) for label, actions in sequences]

    d_s = step_density(h_s, len(raw))
    d_e = None if h_e is None else step_density(h_e, len(raw))
    intervals = [] if d_e is None else dense_intervals(d_s, d_e)
    strip, to_x = density_strip(d_s, d_e, intervals, len(panels) * SIZE)
    frames = []
    # Frame i shows the state after i actions, so its time is i * CONTROL_DT; panels that finish early keep their last frame.
    for i in range(max(len(images) for _, images, _ in panels)):
        time = i * CONTROL_DT
        row = []
        for label, images, success in panels:
            image = images[min(i, len(images) - 1)].copy()
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if i >= len(images) - 1:
                cv2.putText(image, f"success: {success}", (10, SIZE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            row.append(image)
        bar = strip.copy()
        cv2.line(bar, (to_x(time), 0), (to_x(time), STRIP_H - 1), (0, 0, 0), 2)
        inside = [n for n, (a, b) in enumerate(intervals, 1) if a <= i < b]
        stamp = f"t = {time:6.2f} s" + ("   (hold)" if i >= len(raw) else "") + (f"   interval #{inside[0]}" if inside else "")
        cv2.putText(bar, stamp, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        frames.append(np.concatenate([np.concatenate(row, axis=1), bar], axis=0))
    out = args.out_dir / f"{args.demo}_{args.tag}.mp4"
    imageio.mimwrite(out, frames, fps=20, macro_block_size=1)
    print(f"Wrote {out}: " + ", ".join(f"{label}: success {success}" for label, _, success in panels))


if __name__ == "__main__":
    main()
