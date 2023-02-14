from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, look_at, vectorize_pose


@register_env("Reach-v0", max_episode_steps=200)
class ReachEnv(BaseEnv):
    tcp: sapien.Link  # Tool Center Point of the robot
    goal_thresh = 0.05

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.goal_site = None  # create upon rendering

    def _configure_agent(self):
        self._agent_cfg = Panda.get_default_config()

    def _load_agent(self):
        self.agent = Panda(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        )

    def _initialize_agent(self):
        # fmt: off
        # EE at [0.615, 0, 0.17]
        qpos = np.array(
            [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
        )
        # fmt: on
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(Pose([-0.615, 0, 0]))

    def _initialize_task(self):
        low, high = [-0.05, -0.1, 0.05], [0.05, 0.1, 0.3]
        goal_pos = self._episode_rng.uniform(low, high)
        self.goal_pos = np.float32(goal_pos)

        self.init_tcp_pos = self.tcp.pose.p
        self.in_place_margin = np.linalg.norm(self.goal_pos - self.init_tcp_pos)

        if self.goal_site is not None:
            self.goal_site.set_pose(Pose(goal_pos))

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs

    def _get_obs_extra(self) -> OrderedDict:
        return OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos.copy(),
        )

    def evaluate(self, **kwargs) -> dict:
        tcp_pos = self.tcp.pose.p
        goal_pos = self.goal_pos
        tcp_to_goal_dist = np.linalg.norm(goal_pos - tcp_pos)
        success = tcp_to_goal_dist <= self.goal_thresh
        return dict(success=success, tcp_to_goal_dist=tcp_to_goal_dist)

    def compute_dense_reward(self, info, **kwargs):
        tcp_to_goal_dist = info["tcp_to_goal_dist"]
        in_place = tolerance(
            tcp_to_goal_dist,
            bounds=(0, self.goal_thresh),
            margin=self.in_place_margin,
            sigmoid="long_tail",
        )
        reward = 10 * in_place
        return reward

    def _register_cameras(self):
        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

    def _register_render_cameras(self):
        pose = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            if self.goal_site is None:
                self.goal_site = self._build_sphere_site(0.05)
                self.goal_site.set_pose(Pose(self.goal_pos))
            self.goal_site.unhide_visual()
            ret = super().render(mode=mode)
            self.goal_site.hide_visual()
        else:
            ret = super().render(mode=mode)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.
    See also: https://github.com/Farama-Foundation/Metaworld/blob/a98086ababc81560772e27e7f63fe5d120c4cc50/metaworld/envs/reward_utils.py#L10
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, "
                "got {}.".format(value_at_1)
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be strictly between 0 and 1, "
                "got {}.".format(value_at_1)
            )

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError("Unknown sigmoid type {!r}.".format(sigmoid))


def tolerance(
    x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.1
):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
    See also: https://github.com/Farama-Foundation/Metaworld/blob/a98086ababc81560772e27e7f63fe5d120c4cc50/metaworld/envs/reward_utils.py#L76
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError(
            "`margin` must be non-negative. Current value: {}".format(margin)
        )

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if np.isscalar(x) else value
