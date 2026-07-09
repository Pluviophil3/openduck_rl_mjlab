"""OpenDuck Mini V2 flat tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import OPEN_DUCK_ACTION_SCALE, get_open_duck_robot_cfg
import src.tasks.tracking.mdp as tracking_mdp
from src.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from .randomization import (
  RandomizationProfileName,
  apply_open_duck_randomization,
)

OPEN_DUCK_TRACKED_BODY_NAMES = (
  "trunk_assembly",
  "left_roll_to_pitch_assembly",
  "knee_and_ankle_assembly_2",
  "foot_assembly",
  "right_roll_to_pitch_assembly",
  "knee_and_ankle_assembly_4",
  "foot_assembly_2",
  "head_assembly",
)


def open_duck_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  randomization_profile: RandomizationProfileName = "baseline",
) -> ManagerBasedRlEnvCfg:
  """Create the OpenDuck Mini V2 motion-tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_open_duck_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = OPEN_DUCK_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "trunk_assembly"
  motion_cmd.body_names = OPEN_DUCK_TRACKED_BODY_NAMES

  # Scale reference-state initialization perturbations for the 22 cm robot.
  motion_cmd.pose_range = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (-0.005, 0.005),
    "roll": (-0.05, 0.05),
    "pitch": (-0.05, 0.05),
    "yaw": (-0.1, 0.1),
  }
  motion_cmd.velocity_range = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.25, 0.25),
    "pitch": (-0.25, 0.25),
    "yaw": (-0.4, 0.4),
  }
  motion_cmd.joint_position_range = (-0.05, 0.05)

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot_bottom_tpu$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk_assembly",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.005, 0.005),
    1: (-0.005, 0.005),
    2: (-0.005, 0.005),
  }
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.25, 0.25),
    "pitch": (-0.25, 0.25),
    "yaw": (-0.4, 0.4),
  }

  cfg.rewards["motion_global_root_pos"].params["std"] = 0.08
  cfg.rewards["motion_body_pos"].params["std"] = 0.08
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(r"^(?!.*_backlash$).*",)
  )

  cfg.observations["actor"].terms["joint_pos"].func = (
    tracking_mdp.effective_non_backlash_joint_pos_rel
  )
  cfg.observations["actor"].terms["joint_vel"].func = (
    tracking_mdp.effective_non_backlash_joint_vel_rel
  )
  cfg.observations["critic"].terms["joint_pos"].func = (
    tracking_mdp.effective_non_backlash_joint_pos_rel
  )
  cfg.observations["critic"].terms["joint_vel"].func = (
    tracking_mdp.effective_non_backlash_joint_vel_rel
  )

  cfg.terminations["anchor_pos"].params["threshold"] = 0.08
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.08
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "foot_assembly",
    "foot_assembly_2",
    "head_assembly",
  )

  cfg.viewer.body_name = "trunk_assembly"
  cfg.viewer.distance = 0.8

  # The smaller mesh/contact model requires more generous contact buffers.
  cfg.sim.njmax = 640
  cfg.sim.nconmax = 48
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 50

  if not has_state_estimation:
    new_actor_terms = {
      name: term
      for name, term in cfg.observations["actor"].terms.items()
      if name not in ("motion_anchor_pos_b", "base_lin_vel")
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    motion_cmd.sampling_mode = "start"

  apply_open_duck_randomization(
    cfg, "nominal" if play else randomization_profile
  )

  return cfg
