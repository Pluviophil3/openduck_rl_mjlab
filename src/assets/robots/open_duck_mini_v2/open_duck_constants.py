"""OpenDuck Mini V2 robot configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

from src import SRC_PATH

OPEN_DUCK_XML: Path = (
  SRC_PATH
  / "assets"
  / "robots"
  / "open_duck_mini_v2"
  / "xmls"
  / "open_duck_mini_v2_real.xml"
)
assert OPEN_DUCK_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, OPEN_DUCK_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(OPEN_DUCK_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


OPEN_DUCK_LEG_ACTUATOR = XmlPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw",
    ".*_hip_roll",
    ".*_hip_pitch",
    ".*_knee",
    ".*_ankle",
  ),
)

OPEN_DUCK_HEAD_ACTUATOR = XmlPositionActuatorCfg(
  target_names_expr=(
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
  ),
)

OPEN_DUCK_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(OPEN_DUCK_LEG_ACTUATOR, OPEN_DUCK_HEAD_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

OPEN_DUCK_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.22),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

# Preserve the action scaling used by the source OpenDuck environment.
OPEN_DUCK_ACTION_SCALE: dict[str, float] = {
  ".*_hip_yaw": 0.13,
  ".*_hip_roll": 0.13,
  ".*_hip_pitch": 0.13,
  ".*_knee": 0.13,
  ".*_ankle": 0.13,
  "neck_pitch": 0.10,
  "head_pitch": 0.10,
  "head_yaw": 0.10,
  "head_roll": 0.10,
}


def get_open_duck_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=OPEN_DUCK_HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=OPEN_DUCK_ARTICULATION,
  )
