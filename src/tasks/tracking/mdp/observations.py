from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _effective_non_backlash_joint_values(
  env: ManagerBasedRlEnv,
  joint_values: torch.Tensor,
  entity_name: str,
) -> torch.Tensor:
  asset = env.scene[entity_name]
  joint_names = tuple(asset.joint_names)
  backlash_index_by_main = {
    name.removesuffix("_backlash"): index
    for index, name in enumerate(joint_names)
    if name.endswith("_backlash")
  }
  main_joint_ids: list[int] = []
  backlash_joint_ids: list[int] = []
  has_backlash_joint: list[bool] = []
  for index, name in enumerate(joint_names):
    if name.endswith("_backlash"):
      continue
    main_joint_ids.append(index)
    backlash_index = backlash_index_by_main.get(name, index)
    backlash_joint_ids.append(backlash_index)
    has_backlash_joint.append(backlash_index != index)

  main_ids = torch.tensor(main_joint_ids, dtype=torch.long, device=joint_values.device)
  backlash_ids = torch.tensor(
    backlash_joint_ids, dtype=torch.long, device=joint_values.device
  )
  has_backlash = torch.tensor(
    has_backlash_joint, dtype=torch.bool, device=joint_values.device
  )
  main_values = joint_values[:, main_ids]
  backlash_values = joint_values[:, backlash_ids]
  return main_values + torch.where(
    has_backlash, backlash_values, torch.zeros_like(backlash_values)
  )


def _joint_bias(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor | None:
  asset = env.scene[entity_name]
  for attr_name in (
    "joint_pos_bias",
    "joint_position_bias",
    "encoder_bias",
    "joint_encoder_bias",
  ):
    if hasattr(asset.data, attr_name):
      return getattr(asset.data, attr_name)
  return None


def effective_non_backlash_joint_pos_rel(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
  biased: bool = False,
) -> torch.Tensor:
  asset = env.scene[entity_name]
  joint_pos = asset.data.joint_pos
  if biased:
    bias = _joint_bias(env, entity_name)
    if bias is not None:
      joint_pos = joint_pos + bias

  joint_pos = _effective_non_backlash_joint_values(env, joint_pos, entity_name)
  default_joint_pos = asset.data.default_joint_pos
  if default_joint_pos.dim() == 1:
    default_joint_pos = default_joint_pos.unsqueeze(0)
  default_joint_pos = _effective_non_backlash_joint_values(
    env, default_joint_pos, entity_name
  )
  return joint_pos - default_joint_pos


def effective_non_backlash_joint_vel_rel(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
) -> torch.Tensor:
  asset = env.scene[entity_name]
  return _effective_non_backlash_joint_values(
    env, asset.data.joint_vel, entity_name
  )


def clipped_action_history(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  term = env.action_manager.get_term(action_name)
  if hasattr(term, "clipped_action_history"):
    return term.clipped_action_history
  return torch.cat(
    [
      env.action_manager.action,
      env.action_manager.prev_action,
      env.action_manager.prev_prev_action,
    ],
    dim=-1,
  )


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
