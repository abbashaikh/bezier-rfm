"""H30 window data: load all windows onto the device, split by demo, fit and save the scales (plan.md 9.3)."""

import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from bezier_rfm.flow_matching import spline

KEYS = ("q_pos", "q_rot", "H_pos", "H_rot", "h", "z", "logits", "v0", "gripper", "obs", "demo", "grid", "knot",
        "n_tail", "t_start")


def load_windows(path):
    """Read the windows/ group of a window file into host arrays.

    Args:
        path (str): Window HDF5 file.

    Returns:
        dict: Arrays keyed by KEYS, leading axis N.
    """
    with h5py.File(path, "r") as f:
        return {k: f["windows"][k][()] for k in KEYS}


def val_demos(n_demos, val_ratio=0.02, seed=42):
    """Pick validation demos with the Diffusion Policy rule (get_val_mask).

    Args:
        n_demos (int): Number of demos.
        val_ratio (float): Validation fraction.
        seed (int): RNG seed.

    Returns:
        np.ndarray: Sorted validation demo indices, shape (n_val,).
    """
    n_val = min(max(1, round(n_demos * val_ratio)), n_demos - 1)
    return np.sort(np.random.default_rng(seed=seed).choice(n_demos, size=n_val, replace=False))


def split(w, val):
    """Split window indices by demo.

    Args:
        w (dict): Window arrays.
        val (np.ndarray): Validation demo indices.

    Returns:
        tuple: Training and validation window indices, each shape (n,).
    """
    is_val = np.isin(w["demo"], val)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def save_stats(path, stats, val):
    """Write the scales and the validation demos as JSON.

    Args:
        path (str): Output file, e.g. checkpoints/<run>/stats.json.
        stats (spline.Stats): Scales fitted on the training demos.
        val (np.ndarray): Validation demo indices.
    """
    d = {k: np.asarray(v).tolist() for k, v in stats._asdict().items()}
    Path(path).write_text(json.dumps({"stats": d, "val_demos": [int(v) for v in val]}, indent=1))


def load_stats(path):
    """Read scales and validation demos written by save_stats.

    Args:
        path (str): stats.json file.

    Returns:
        tuple: spline.Stats (float32) and validation demo indices (np.ndarray).
    """
    d = json.loads(Path(path).read_text())
    stats = spline.Stats(**{k: jnp.asarray(v, jnp.float32) for k, v in d["stats"].items()})
    return stats, np.array(d["val_demos"])


def device_data(w, stats):
    """Put the targets and conditioning of all windows on the default device.

    Args:
        w (dict): Window arrays.
        stats (spline.Stats): Scales.

    Returns:
        tuple: Targets (spline.State) and conditioning (spline.Cond), leading axis N, float32.
    """
    return jax.device_put((spline.target_state(w), spline.make_cond(w, stats)))


def take(tree, idx):
    """Gather rows of a pytree of arrays.

    Args:
        tree (pytree): Arrays with a shared leading axis.
        idx (jnp.ndarray): Row indices, shape (n,).

    Returns:
        pytree: Gathered arrays, leading axis n.
    """
    return jax.tree.map(lambda a: a[idx], tree)
