"""Find the clock constant c of the S rule with angular speed (H35) that gives the same mean K as linear speed only at c_ref."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from bezier_rfm.fitting.density import speed_weight
from bezier_rfm.fitting.knots import clock_h


def mean_k(weights, c, h_min):
    """Return the mean segment count of the S rule over the demos.

    Args:
        weights (list): Weight per action of each demo, arrays of shape (N,).
        c (float): Clock constant.
        h_min (float): Shortest segment, s.

    Returns:
        float: Mean K over the demos.
    """
    return float(np.mean([len(clock_h(w, c, h_min)) for w in weights]))


def main():
    """Bisect c for the given lever lengths and print it with the mean K.

    Args:
        None: Arguments come from the command line.

    Returns:
        None: Nothing.
    """
    root = Path("data/robomimic/datasets/tool_hang/ph")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=root / "low_dim_abs_q0start.hdf5")
    parser.add_argument("--c-ref", type=float, default=3.10, help="Clock constant of the linear-only reference")
    parser.add_argument("--lengths", type=float, nargs="+", default=[0.01, 0.165, 0.10],
                        help="Lever lengths L (m), empty gripper then each held object")
    parser.add_argument("--h-min", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()

    with h5py.File(args.actions, "r") as f:
        actions = [f[f"data/{name}/actions"][:] for name in f["data"].keys()]
    target = mean_k([speed_weight(a) for a in actions], args.c_ref, args.h_min)
    weights = [speed_weight(a, lengths=args.lengths) for a in actions]
    # v_eff >= v gives a smaller weight, so the matched c is below c_ref; mean K does not increase with c.
    lo, hi = args.c_ref / 10, args.c_ref
    for _ in range(args.iterations):
        mid = 0.5 * (lo + hi)
        if mean_k(weights, mid, args.h_min) >= target:
            lo = mid
        else:
            hi = mid
    print(f"{len(actions)} demos, reference c {args.c_ref:g}: mean K {target:.3f}")
    print(f"lengths {args.lengths}: c {lo:.4f}, mean K {mean_k(weights, lo, args.h_min):.3f}")


if __name__ == "__main__":
    main()
