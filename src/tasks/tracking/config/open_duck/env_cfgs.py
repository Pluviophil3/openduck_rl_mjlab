"""OpenDuck Mini V2 flat tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.assets.robots import OPEN_DUCK_ACTION_SCALE, get_open_duck_robot_cfg
import src.tasks.tracking.mdp as tracking_mdp
from src.tasks.tracking.mdp import MotionCommandCfg, OpenDuckJointPositionActionCfg
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


def _reorder_terms(
  terms: dict[str, ObservationTermCfg],
  names: tuple[str, ...],
) -> dict[str, ObservationTermCfg]:
  return {name: terms[name] for name in names if name in terms}


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
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=("left_foot_bottom_tpu", "right_foot_bottom_tpu"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (self_collision_cfg, feet_ground_cfg)

  joint_pos_action = OpenDuckJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=OPEN_DUCK_ACTION_SCALE,
    use_default_offset=True,
  )
  cfg.actions["joint_pos"] = joint_pos_action
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

  cfg.observations["actor"].terms.pop("motion_anchor_ori_b", None)
  cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
    func=tracking_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
  )
  cfg.observations["actor"].terms["actions"] = ObservationTermCfg(
    func=tracking_mdp.clipped_action_history,
    params={"action_name": "joint_pos"},
  )
  cfg.observations["actor"].terms["feet_contact"] = ObservationTermCfg(
    func=tracking_mdp.foot_contact,
    params={"sensor_name": feet_ground_cfg.name},
  )

  cfg.observations["critic"].terms["base_lin_acc"] = ObservationTermCfg(
    func=tracking_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
  )
  cfg.observations["critic"].terms["actions"] = ObservationTermCfg(
    func=tracking_mdp.clipped_action_history,
    params={"action_name": "joint_pos"},
  )
  cfg.observations["critic"].terms["feet_contact"] = ObservationTermCfg(
    func=tracking_mdp.foot_contact,
    params={"sensor_name": feet_ground_cfg.name},
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

  cfg.observations["actor"].terms = _reorder_terms(
    cfg.observations["actor"].terms,
    (
      "command",
      "motion_anchor_pos_b",
      "base_lin_vel",
      "base_ang_vel",
      "base_lin_acc",
      "joint_pos",
      "joint_vel",
      "actions",
      "feet_contact",
    ),
  )
  cfg.observations["critic"].terms = _reorder_terms(
    cfg.observations["critic"].terms,
    (
      "command",
      "motion_anchor_pos_b",
      "motion_anchor_ori_b",
      "body_pos",
      "body_ori",
      "base_lin_vel",
      "base_ang_vel",
      "base_lin_acc",
      "joint_pos",
      "joint_vel",
      "actions",
      "feet_contact",
    ),
  )

  if play:
    cfg.episode_length_s = int(1e9)
    motion_cmd.sampling_mode = "start"

  apply_open_duck_randomization(
    cfg, "nominal" if play else randomization_profile
  )

  return cfg
