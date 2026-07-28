"""Composable domain-randomization profiles for OpenDuck Mini."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.actuator import DelayedActuator, DelayedActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, push_by_setting_velocity
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import (
  NoiseModelWithAdditiveBiasCfg,
  UniformNoiseCfg,
)

from src.tasks.tracking.mdp import MotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


RandomizationProfileName = Literal[
  "nominal",
  "baseline",
  "sensor",
  "dynamics",
  "actuator",
  "latency",
  "light",
  "full",
]

RANDOMIZATION_PROFILE_NAMES: tuple[RandomizationProfileName, ...] = (
  "nominal",
  "baseline",
  "sensor",
  "dynamics",
  "actuator",
  "latency",
  "light",
  "full",
)

_RANDOMIZATION_EVENT_NAMES = {
  "push_robot",
  "base_com",
  "encoder_bias",
  "foot_friction",
  "body_mass",
  "joint_damping",
  "joint_friction",
  "joint_armature",
  "pd_gains",
  "effort_limits",
}

_MASS_BODY_NAMES = (
  "trunk_assembly",
  "hip_roll_assembly",
  "left_roll_to_pitch_assembly",
  "knee_and_ankle_assembly",
  "knee_and_ankle_assembly_2",
  "foot_assembly",
  "neck_pitch_assembly",
  "head_pitch_to_yaw",
  "neck_yaw_assembly",
  "head_assembly",
  "left_antenna_holder",
  "right_antenna_holder",
  "hip_roll_assembly_2",
  "right_roll_to_pitch_assembly",
  "knee_and_ankle_assembly_3",
  "knee_and_ankle_assembly_4",
  "foot_assembly_2",
)

_PUSH_VELOCITY_RANGE = {
  "x": (-0.2, 0.2),
  "y": (-0.2, 0.2),
  "z": (-0.1, 0.1),
  "roll": (-0.25, 0.25),
  "pitch": (-0.25, 0.25),
  "yaw": (-0.4, 0.4),
}

_RESET_POSE_RANGE = {
  "x": (-0.02, 0.02),
  "y": (-0.02, 0.02),
  "z": (-0.005, 0.005),
  "roll": (-0.05, 0.05),
  "pitch": (-0.05, 0.05),
  "yaw": (-0.1, 0.1),
}


@dataclass(frozen=True)
class OpenDuckRandomizationProfile:
  """Ranges are multiplicative unless their field name says otherwise."""

  observation_corruption: bool = False
  reset_perturbation: bool = False
  push_robot: bool = False
  encoder_bias_rad: tuple[float, float] | None = None
  gyro_bias_rad_s: tuple[float, float] | None = None
  foot_friction: tuple[float, float] | None = None
  trunk_com_offset_m: float | None = None
  body_mass_scale: tuple[float, float] | None = None
  joint_damping_scale: tuple[float, float] | None = None
  joint_friction_scale: tuple[float, float] | None = None
  joint_armature_scale: tuple[float, float] | None = None
  kp_scale: tuple[float, float] | None = None
  effort_scale: tuple[float, float] | None = None
  action_delay_control_steps: tuple[int, int] = (0, 0)
  observation_delay_control_steps: tuple[int, int] = (0, 0)


PROFILES: dict[RandomizationProfileName, OpenDuckRandomizationProfile] = {
  "nominal": OpenDuckRandomizationProfile(),
  # Reproduces the randomization used to train models/Sway_t2.
  "baseline": OpenDuckRandomizationProfile(
    observation_corruption=True,
    reset_perturbation=True,
    push_robot=True,
    encoder_bias_rad=(-0.01, 0.01),
    foot_friction=(0.3, 1.2),
    trunk_com_offset_m=0.005,
  ),
  "sensor": OpenDuckRandomizationProfile(
    observation_corruption=True,
    encoder_bias_rad=(-0.01, 0.01),
    gyro_bias_rad_s=(-0.03, 0.03),
  ),
  "dynamics": OpenDuckRandomizationProfile(
    foot_friction=(0.5, 1.0),
    trunk_com_offset_m=0.003,
    body_mass_scale=(0.95, 1.05),
    joint_damping_scale=(0.9, 1.1),
    joint_friction_scale=(0.9, 1.1),
    joint_armature_scale=(0.9, 1.1),
  ),
  "actuator": OpenDuckRandomizationProfile(
    kp_scale=(0.9, 1.1),
    effort_scale=(0.9, 1.0),
  ),
  "latency": OpenDuckRandomizationProfile(
    action_delay_control_steps=(0, 2),
    observation_delay_control_steps=(0, 2),
  ),
  "light": OpenDuckRandomizationProfile(
    observation_corruption=True,
    reset_perturbation=True,
    encoder_bias_rad=(-0.01, 0.01),
    gyro_bias_rad_s=(-0.03, 0.03),
    foot_friction=(0.5, 1.0),
    trunk_com_offset_m=0.003,
    body_mass_scale=(0.95, 1.05),
    joint_damping_scale=(0.9, 1.1),
    joint_friction_scale=(0.9, 1.1),
    joint_armature_scale=(0.9, 1.1),
    kp_scale=(0.9, 1.1),
    effort_scale=(0.9, 1.0),
    action_delay_control_steps=(0, 1),
    observation_delay_control_steps=(0, 1),
  ),
  "full": OpenDuckRandomizationProfile(
    observation_corruption=True,
    reset_perturbation=True,
    push_robot=True,
    encoder_bias_rad=(-0.01, 0.01),
    gyro_bias_rad_s=(-0.03, 0.03),
    foot_friction=(0.3, 1.2),
    trunk_com_offset_m=0.005,
    body_mass_scale=(0.9, 1.1),
    joint_damping_scale=(0.8, 1.2),
    joint_friction_scale=(0.8, 1.2),
    joint_armature_scale=(0.8, 1.2),
    kp_scale=(0.85, 1.15),
    effort_scale=(0.85, 1.05),
    action_delay_control_steps=(0, 2),
    observation_delay_control_steps=(0, 2),
  ),
}


@requires_model_fields("actuator_forcerange")
def _effort_limits_with_delayed_actuators(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Scale effort limits for plain or delay-wrapped actuator groups.

  mjlab 1.2.0's built-in ``dr.effort_limits`` does not unwrap
  ``DelayedActuator`` even though ``dr.pd_gains`` does.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  actuator_ids = asset_cfg.actuator_ids
  if isinstance(actuator_ids, slice):
    actuators = asset.actuators[actuator_ids]
  elif isinstance(actuator_ids, list):
    actuators = [asset.actuators[i] for i in actuator_ids]
  else:
    actuators = [asset.actuators[actuator_ids]]

  default_forcerange = env.sim.get_default_field("actuator_forcerange")
  for actuator in actuators:
    if isinstance(actuator, DelayedActuator):
      actuator = actuator.base_actuator
    ctrl_ids = actuator.global_ctrl_ids
    scales = torch.empty(
      (len(env_ids), len(ctrl_ids)), device=env.device
    ).uniform_(*effort_limit_range)
    env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
      default_forcerange[ctrl_ids, 0] * scales
    )
    env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
      default_forcerange[ctrl_ids, 1] * scales
    )


def profile_as_dict(name: RandomizationProfileName) -> dict[str, object]:
  """Return a serializable profile description for manifests and reports."""
  return {"name": name, **asdict(PROFILES[name])}


def apply_open_duck_randomization(
  cfg: ManagerBasedRlEnvCfg,
  profile_name: RandomizationProfileName,
) -> None:
  """Mutate an OpenDuck environment config to use one named profile."""
  if profile_name not in PROFILES:
    raise ValueError(
      f"Unknown OpenDuck randomization profile {profile_name!r}; "
      f"expected one of {RANDOMIZATION_PROFILE_NAMES}"
    )
  profile = PROFILES[profile_name]

  for event_name in _RANDOMIZATION_EVENT_NAMES:
    cfg.events.pop(event_name, None)

  actor = cfg.observations["actor"]
  actor.enable_corruption = profile.observation_corruption
  actor.terms["base_ang_vel"].noise = UniformNoiseCfg(
    n_min=-0.2, n_max=0.2
  )
  if profile.gyro_bias_rad_s is not None:
    actor.terms["base_ang_vel"].noise = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
      bias_noise_cfg=UniformNoiseCfg(
        n_min=profile.gyro_bias_rad_s[0],
        n_max=profile.gyro_bias_rad_s[1],
      ),
    )
  obs_delay_min, obs_delay_max = profile.observation_delay_control_steps
  delayed_observation_terms = ("base_ang_vel", "joint_pos", "joint_vel")
  for term_name, term in actor.terms.items():
    if term_name in delayed_observation_terms:
      term.delay_min_lag = obs_delay_min
      term.delay_max_lag = obs_delay_max
      term.delay_hold_prob = 0.8 if obs_delay_max > 0 else 0.0
    else:
      term.delay_min_lag = 0
      term.delay_max_lag = 0
      term.delay_hold_prob = 0.0

  motion_cmd = cast(MotionCommandCfg, cfg.commands["motion"])
  if profile.reset_perturbation:
    motion_cmd.pose_range = dict(_RESET_POSE_RANGE)
    motion_cmd.velocity_range = dict(_PUSH_VELOCITY_RANGE)
    motion_cmd.joint_position_range = (-0.05, 0.05)
  else:
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

  if profile.push_robot:
    cfg.events["push_robot"] = EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": dict(_PUSH_VELOCITY_RANGE)},
    )

  if profile.trunk_com_offset_m is not None:
    offset = profile.trunk_com_offset_m
    cfg.events["base_com"] = EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=("trunk_assembly",)
        ),
        "operation": "add",
        "ranges": {axis: (-offset, offset) for axis in range(3)},
      },
    )

  if profile.encoder_bias_rad is not None:
    cfg.events["encoder_bias"] = EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": profile.encoder_bias_rad,
      },
    )

  if profile.foot_friction is not None:
    cfg.events["foot_friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot_bottom_tpu$"
        ),
        "operation": "abs",
        "ranges": profile.foot_friction,
        "shared_random": True,
      },
    )

  if profile.body_mass_scale is not None:
    alpha_range = tuple(
      0.5 * math.log(scale) for scale in profile.body_mass_scale
    )
    cfg.events["body_mass"] = EventTermCfg(
      # Scale mass and inertia together instead of changing mass alone.
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=_MASS_BODY_NAMES),
        "alpha_range": alpha_range,
      },
    )
  if profile.joint_damping_scale is not None:
    cfg.events["joint_damping"] = EventTermCfg(
      func=dr.joint_damping,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": profile.joint_damping_scale,
      },
    )
  if profile.joint_friction_scale is not None:
    cfg.events["joint_friction"] = EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": profile.joint_friction_scale,
      },
    )
  if profile.joint_armature_scale is not None:
    cfg.events["joint_armature"] = EventTermCfg(
      func=dr.joint_armature,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": profile.joint_armature_scale,
      },
    )
  if profile.kp_scale is not None:
    cfg.events["pd_gains"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        # DR operates on the two configured actuator groups (legs and head),
        # rather than the 14 individual MuJoCo controls.
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1]),
        "operation": "scale",
        "kp_range": profile.kp_scale,
        # XML position actuators currently have kv=0. Keep it unchanged.
        "kd_range": (1.0, 1.0),
      },
    )
  if profile.effort_scale is not None:
    cfg.events["effort_limits"] = EventTermCfg(
      func=_effort_limits_with_delayed_actuators,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1]),
        "effort_limit_range": profile.effort_scale,
      },
    )

  robot_cfg = cfg.scene.entities["robot"]
  if robot_cfg.articulation is None:
    raise ValueError("OpenDuck robot must have articulation information")
  base_actuators = tuple(
    actuator.base_cfg if isinstance(actuator, DelayedActuatorCfg) else actuator
    for actuator in robot_cfg.articulation.actuators
  )
  delay_min, delay_max = profile.action_delay_control_steps
  if delay_max > 0:
    physics_lag_min = delay_min * cfg.decimation
    physics_lag_max = delay_max * cfg.decimation
    robot_cfg.articulation.actuators = tuple(
      DelayedActuatorCfg(
        base_cfg=actuator,
        delay_target="position",
        delay_min_lag=physics_lag_min,
        delay_max_lag=physics_lag_max,
        delay_hold_prob=0.8,
      )
      for actuator in base_actuators
    )
  else:
    robot_cfg.articulation.actuators = base_actuators
