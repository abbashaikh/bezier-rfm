"""Capture per-step diagnostic traces of Lift and Tool Hang demonstrations and command replays."""

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bezier_rfm.datasets.robomimic import arm_controller, load_demo_model

CONTROL_DT = 0.05
# Per-state fields: name -> shape of one row.
STATE_FIELDS = {
    "sim_time": (),
    "state": (32,),
    "ach_pos": (3,),
    "ach_rot": (3, 3),
    "ach_linvel": (3,),
    "ach_angvel": (3,),
    "finger_q": (2,),
    "cube_pos": (3,),
    "cube_rot": (3, 3),
    "gripper_to_cube": (),
    "cube_height": (),
    "contact": (),
    "success": (),
}
# Per-action fields: name -> shape of one row.
ACTION_FIELDS = {
    "t_start": (),
    "t_end": (),
    "cmd_pos": (3,),
    "cmd_rot": (3, 3),
    "gripper_cmd": (),
    "goal_pos": (3,),
    "goal_rot": (3, 3),
}
BOOL_FIELDS = ("contact", "success", "valid")
FINGER_Q = slice(8, 10)
# Tool Hang records one pose per object instead of the Lift cube fields.
TOOL_HANG_OBJECTS = ("stand", "frame", "tool")
PHASES = ["first", "approach", "grasp", "lift", "final"]


def is_tool_hang(env):
    """Return whether the environment is Tool Hang.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        bool: True for Tool Hang, False for Lift.
    """
    return type(env.env).__name__ == "ToolHang"


def state_fields(env):
    """Return the per-state fields of the environment's task.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        dict: Field name -> shape of one row.
    """
    if not is_tool_hang(env):
        return STATE_FIELDS
    fields = {k: v for k, v in STATE_FIELDS.items() if not k.startswith(("cube", "gripper_to_cube"))}
    model = env.env.sim.model
    # A flattened MuJoCo state is [time, qpos, qvel].
    fields["state"] = (1 + model.nq + model.nv,)
    for name in TOOL_HANG_OBJECTS:
        fields[f"{name}_pos"] = (3,)
        fields[f"{name}_rot"] = (3, 3)
    return fields


def read_state(env):
    """Read the grip-site pose and velocity, fingers, objects, contact, and success.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        dict: One row of every entry of STATE_FIELDS.
    """
    base = env.env
    sim = base.sim
    # After mj_step, poses and velocities describe the state before the last integration.
    sim.forward()
    model, data = sim.model._model, sim.data._data
    site = base.robots[0].eef_site_id["right"]
    velocity = np.zeros(6)
    # World-frame [angular, linear] velocity of the grip site.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
    state = sim.get_state().flatten().copy()
    pos = data.site_xpos[site].copy()
    row = {
        "sim_time": float(data.time),
        "ach_pos": pos,
        "ach_rot": data.site_xmat[site].reshape(3, 3).copy(),
        "ach_linvel": velocity[3:].copy(),
        "ach_angvel": velocity[:3].copy(),
        "state": state,
        "finger_q": state[FINGER_Q].copy(),
        "success": bool(base._check_success()),
    }
    gripper = base.robots[0].gripper["right"]
    if is_tool_hang(env):
        for name in TOOL_HANG_OBJECTS:
            body = base.obj_body_id[name]
            row[f"{name}_pos"] = data.xpos[body].copy()
            row[f"{name}_rot"] = data.xmat[body].reshape(3, 3).copy()
        row["contact"] = any(base.check_contact(gripper, getattr(base, name)) for name in TOOL_HANG_OBJECTS)
        return row
    cube_pos = data.xpos[base.cube_body_id].copy()
    return {
        **row,
        "cube_pos": cube_pos,
        "cube_rot": data.xmat[base.cube_body_id].reshape(3, 3).copy(),
        "gripper_to_cube": float(np.linalg.norm(pos - cube_pos)),
        # Lift counts the cube as lifted when its centre is 0.04 m above the table top.
        "cube_height": float(cube_pos[2] - base.model.mujoco_arena.table_offset[2]),
        "contact": bool(base.check_contact(gripper, base.cube)),
    }


def empty_trace(n, fields=STATE_FIELDS):
    """Create a trace for n actions, filled with NaN and False.

    Args:
        n (int): Number of actions.
        fields (dict): Per-state fields, name -> shape of one row.

    Returns:
        dict: Arrays with n + 1 state rows and n action rows.
    """
    trace = {}
    for table, rows in ((fields, n + 1), (ACTION_FIELDS, n)):
        for name, shape in table.items():
            trace[name] = np.zeros((rows, *shape), bool) if name in BOOL_FIELDS else np.full((rows, *shape), np.nan)
    trace["valid"] = np.zeros(n + 1, bool)
    trace["t_start"] = CONTROL_DT * np.arange(n)
    trace["t_end"] = CONTROL_DT * np.arange(1, n + 1)
    return trace


def set_commands(trace, actions):
    """Write commanded poses and gripper commands from 7-number absolute actions.

    Args:
        trace (dict): Trace to fill.
        actions (np.ndarray): Absolute actions, shape (N, 7).

    Returns:
        None: Nothing.
    """
    trace["cmd_pos"][:] = actions[:, :3]
    trace["cmd_rot"][:] = Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
    trace["gripper_cmd"][:] = actions[:, 6]


def set_state_row(trace, k, row):
    """Write one state row and mark it valid.

    Args:
        trace (dict): Trace to fill.
        k (int): State index.
        row (dict): Output of read_state.

    Returns:
        None: Nothing.
    """
    for name, value in row.items():
        trace[name][k] = value
    trace["valid"][k] = True



def record_trace(env, demo, actions):
    """Build the trace of the recorded demonstration from its stored states.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (h5py.Group): One stored demonstration.
        actions (np.ndarray): Recorded absolute actions, shape (N, 7).

    Returns:
        dict: Trace with N valid states. State N does not exist and stays invalid.
    """
    states = demo["states"][:]
    load_demo_model(env, demo)
    trace = empty_trace(len(actions), state_fields(env))
    set_commands(trace, actions)
    for k, state in enumerate(states):
        env.reset_to({"states": state})
        set_state_row(trace, k, read_state(env))
    return trace


def evaluate_commands(env, demo, actions, start="refreshed"):
    """Restore s_0 once, run absolute actions in sequence, and build the trace.

    Args:
        env (EnvRobosuite): Environment with an absolute OSC_POSE controller.
        demo (h5py.Group): Demonstration that supplies the model and s_0.
        actions (np.ndarray): Absolute actions, shape (N, 7).
        start (str): "refreshed" or "stale" first step.

    Returns:
        dict: Trace with N + 1 valid states and the controller goal of every action.
    """
    load_demo_model(env, demo)
    trace = empty_trace(len(actions), state_fields(env))
    set_commands(trace, actions)
    if start == "refreshed":
        # After a model load, new_update stays False, so the first step would use a pose from before reset_to.
        arm_controller(env).new_update = True
    set_state_row(trace, 0, read_state(env))
    for t, action in enumerate(actions):
        env.step(action)
        controller = arm_controller(env)
        trace["goal_pos"][t] = controller.goal_pos
        trace["goal_rot"][t] = controller.goal_ori
        # Action t runs from state t to state t + 1.
        set_state_row(trace, t + 1, read_state(env))
    return trace

def phase_labels(recording, n):
    """Label each action by the task phase of the recording.

    Args:
        recording (dict): Recorded frames or trace, with contact and success per state.
        n (int): Number of actions.

    Returns:
        np.ndarray: One entry of PHASES per action, shape (n,).
    """
    contact = int(np.argmax(recording["contact"])) if recording["contact"].any() else n
    lifted = int(np.argmax(recording["success"])) if recording["success"].any() else n
    labels = []
    for t in range(n):
        # Action t is compared with recorded state t + 1.
        if t == 0:
            labels.append("first")
        elif t == n - 1:
            labels.append("final")
        elif t + 1 < contact:
            labels.append("approach")
        elif t + 1 < lifted:
            labels.append("grasp")
        else:
            labels.append("lift")
    return np.array(labels)


def geodesic(rot_a, rot_b):
    """Return the angle of R_a^T R_b for stacks of rotation matrices.

    Args:
        rot_a (np.ndarray): Rotation matrices, shape (M, 3, 3).
        rot_b (np.ndarray): Rotation matrices, shape (M, 3, 3).

    Returns:
        np.ndarray: Angles in radians, NaN where an input row is NaN, shape (M,).
    """
    out = np.full(len(rot_a), np.nan)
    ok = np.isfinite(rot_a).all(axis=(1, 2)) & np.isfinite(rot_b).all(axis=(1, 2))
    if ok.any():
        out[ok] = Rotation.from_matrix(np.swapaxes(rot_a[ok], 1, 2) @ rot_b[ok]).magnitude()
    return out


def command_fidelity(trace):
    """Return the gap between the controller goal and the command at each action.

    Args:
        trace (dict): Replay trace.

    Returns:
        np.ndarray: Position gap (m) and angle (rad), shape (N, 2).
    """
    pos = np.linalg.norm(trace["goal_pos"] - trace["cmd_pos"], axis=1)
    return np.stack([pos, geodesic(trace["goal_rot"], trace["cmd_rot"])], axis=1)


def tracking_errors(trace):
    """Return e_p(t) = |p_t^cmd - p_{t+1}^ach| and e_R(t) = angle(R_t^cmd^T R_{t+1}^ach).

    Args:
        trace (dict): Recorded or replay trace.

    Returns:
        np.ndarray: Position (m) and angle (rad) per action, NaN where state t + 1 is invalid, shape (N, 2).
    """
    # Invalid state rows hold NaN, so their errors come out as NaN.
    pos = np.linalg.norm(trace["cmd_pos"] - trace["ach_pos"][1:], axis=1)
    return np.stack([pos, geodesic(trace["cmd_rot"], trace["ach_rot"][1:])], axis=1)


ACHIEVED_METRICS = ["eef_pos_m", "eef_rot_rad", "cube_pos_m", "cube_rot_rad", "finger_q_m"]


def achieved_errors(trace, reference):
    """Compare the states after each action between two traces of the same demonstration.

    Args:
        trace (dict): Trace to evaluate.
        reference (dict): Reference trace, usually the recording.

    Returns:
        np.ndarray: One column per entry of ACHIEVED_METRICS, NaN where either state t + 1 is invalid, shape (N, 5).
    """
    n = len(trace["cmd_pos"])
    ok = trace["valid"][1:] & reference["valid"][1:]
    a = {name: trace[name][1:] for name in ("ach_pos", "ach_rot", "cube_pos", "cube_rot", "finger_q")}
    b = {name: reference[name][1:] for name in a}
    out = np.full((n, len(ACHIEVED_METRICS)), np.nan)
    out[ok, 0] = np.linalg.norm(a["ach_pos"][ok] - b["ach_pos"][ok], axis=1)
    out[ok, 1] = geodesic(a["ach_rot"][ok], b["ach_rot"][ok])
    out[ok, 2] = np.linalg.norm(a["cube_pos"][ok] - b["cube_pos"][ok], axis=1)
    out[ok, 3] = geodesic(a["cube_rot"][ok], b["cube_rot"][ok])
    out[ok, 4] = np.abs(a["finger_q"][ok] - b["finger_q"][ok]).max(axis=1)
    return out


def summarize(values, phases, mask=None):
    """Return RMS, median, 95th percentile, and maximum of one error per phase and overall.

    Args:
        values (np.ndarray): One error per action, shape (N,).
        phases (np.ndarray): Phase label per action, shape (N,).
        mask (np.ndarray | None): Actions to include, shape (N,).

    Returns:
        dict: Phase name or "all" -> dict with count, rms, median, p95, and max.
    """
    keep = np.isfinite(values) if mask is None else np.isfinite(values) & mask
    out = {}
    for phase in [*PHASES, "all"]:
        sel = keep & (phases == phase) if phase != "all" else keep
        v = values[sel]
        if len(v):
            out[phase] = {"count": int(len(v)), "rms": float(np.sqrt(np.mean(v**2))), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    return out


def commands_to_actions(cmd_pos, cmd_rot, gripper_cmd):
    """Pack commanded poses and gripper commands into 7-number absolute actions.

    Args:
        cmd_pos (np.ndarray): Commanded positions, shape (N, 3).
        cmd_rot (np.ndarray): Commanded rotation matrices, shape (N, 3, 3).
        gripper_cmd (np.ndarray): Gripper commands, shape (N,).

    Returns:
        np.ndarray: Absolute actions, shape (N, 7).
    """
    return np.concatenate([cmd_pos, Rotation.from_matrix(cmd_rot).as_rotvec(), gripper_cmd[:, None]], axis=1)


def write_trace(group, trace, attrs):
    """Write a trace and its metadata to an HDF5 group.

    Args:
        group (h5py.Group): Empty target group.
        trace (dict): Trace arrays.
        attrs (dict): Metadata; dicts and lists are stored as JSON.

    Returns:
        None: Nothing.
    """
    for name, value in trace.items():
        group.create_dataset(name, data=value)
    for name, value in attrs.items():
        group.attrs[name] = json.dumps(value) if isinstance(value, (dict, list)) else value


def read_trace(group):
    """Read a trace and its metadata from an HDF5 group.

    Args:
        group (h5py.Group): Group written by write_trace.

    Returns:
        tuple: Trace arrays (dict) and metadata (dict).
    """
    return {name: group[name][:] for name in group}, dict(group.attrs)


# Pass-rule limits of lift_diagnostics_baseline.ipynb, section 4.6. The gripper limits are proposals, not measurements.
FIDELITY_TOL = 1e-9
GRIPPER_POS_LIMIT = 3e-3
GRIPPER_ROT_LIMIT = np.radians(1.0)


def pass_rule(trace, reference):
    """Evaluate a candidate replay against the reference replay of the same demonstration (section 4.6).

    Args:
        trace (dict): Candidate replay trace.
        reference (dict): Reference replay trace, usually replay_q0start.

    Returns:
        dict: Per-level values (m, rad), the gate result, and the level 2 and 3 flags.
    """
    n = len(trace["cmd_pos"])
    fidelity = command_fidelity(trace)
    achieved = achieved_errors(trace, reference)
    free = np.isin(phase_labels(reference, n), ["first", "approach"])
    cmd_pos = np.linalg.norm(trace["cmd_pos"] - reference["cmd_pos"], axis=1)
    cmd_rot = geodesic(trace["cmd_rot"], reference["cmd_rot"])
    close = lambda g: int(np.flatnonzero(np.diff(g))[0]) + 1
    out = {
        "fidelity_pos_m": float(fidelity[:, 0].max()),
        "fidelity_rot_rad": float(fidelity[:, 1].max()),
        "cmd_pos_max_m": float(cmd_pos.max()),
        "cmd_rot_max_rad": float(cmd_rot.max()),
        "free_pos_max_m": float(np.nanmax(achieved[free, 0])),
        "free_rot_max_rad": float(np.nanmax(achieved[free, 1])),
        "contact_pos_max_m": float(np.nanmax(achieved[~free, 0])),
        "contact_rot_max_rad": float(np.nanmax(achieved[~free, 1])),
        "cube_pos_max_m": float(np.nanmax(achieved[:, 2])),
        "cube_rot_max_rad": float(np.nanmax(achieved[:, 3])),
        "close_step": close(trace["gripper_cmd"]),
        "ref_close_step": close(reference["gripper_cmd"]),
        "success": bool(trace["success"][-1]),
        "ref_success": bool(reference["success"][-1]),
        "height_diff_m": float(trace["cube_height"][-1] - reference["cube_height"][-1]),
    }
    fidelity_ok = out["fidelity_pos_m"] <= FIDELITY_TOL and out["fidelity_rot_rad"] <= FIDELITY_TOL
    # Option A: the gate is fidelity, the close step, and the lift outcome; levels 2 and 3 are flags to inspect.
    out["gate"] = fidelity_ok and out["close_step"] == out["ref_close_step"] and (out["success"] or not out["ref_success"])
    out["level2_ok"] = bool(out["free_pos_max_m"] <= GRIPPER_POS_LIMIT and out["free_rot_max_rad"] <= GRIPPER_ROT_LIMIT)
    out["level3_ok"] = bool(out["contact_pos_max_m"] <= GRIPPER_POS_LIMIT and out["contact_rot_max_rad"] <= GRIPPER_ROT_LIMIT)
    return out
