"""Load RoboMimic demonstrations into robosuite and read the simulator."""

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def require(condition, message):
    """Stop the script when a hard check fails.

    Args:
        condition (bool): Result of the check.
        message (str): Description of the failed check.

    Returns:
        None: Nothing.
    """
    if not condition:
        raise SystemExit(f"HARD FAILURE: {message}")


def demo_keys(data):
    """Return demonstration names in numeric order.

    Args:
        data (h5py.Group): The HDF5 "data" group.

    Returns:
        list: Names such as "demo_0".
    """
    return sorted(data.keys(), key=lambda key: int(key.split("_")[1]))


def make_env(env_args, obs_keys):
    """Create an environment from stored metadata.

    Args:
        env_args (dict): Environment metadata.
        obs_keys (list): Low-dimensional observation names.

    Returns:
        EnvRobosuite: The environment wrapper.
    """
    # EnvRobosuite.get_observation reads this table, so it must exist first.
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(obs_keys), "rgb": []}})
    return EnvUtils.create_env_from_metadata(env_args, render=False, render_offscreen=False, use_image_obs=False)


def arm_controller(env):
    """Return the arm controller of the single robot.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        object: The OSC part controller.
    """
    robot = env.env.robots[0]
    return robot.part_controllers[robot.arms[0]]


def load_demo_model(env, demo):
    """Load one demonstration's stored MuJoCo model.

    Args:
        env (EnvRobosuite): The environment wrapper.
        demo (h5py.Group): One stored demonstration.

    Returns:
        None: Nothing.
    """
    env.reset_to({"states": demo["states"][0], "model": demo.attrs["model_file"]})
    model = env.env.sim.model
    # A flattened MuJoCo state is [time, qpos, qvel].
    require(len(demo["states"][0]) == 1 + model.nq + model.nv, "stored state does not match the model layout")


def read_result(env):
    """Read the up-to-date full state and grip-site pose.

    Args:
        env (EnvRobosuite): The environment wrapper.

    Returns:
        dict: State, shape (1 + nq + nv,), grip-site position, shape (3,), and rotation matrix, shape (3, 3).
    """
    sim = env.env.sim
    # After mj_step, site poses describe the qpos before the last integration.
    sim.forward()
    site = env.env.robots[0].eef_site_id["right"]
    return {
        "state": sim.get_state().flatten().copy(),
        "pos": sim.data.site_xpos[site].copy(),
        "rot": sim.data.site_xmat[site].reshape(3, 3).copy(),
    }
