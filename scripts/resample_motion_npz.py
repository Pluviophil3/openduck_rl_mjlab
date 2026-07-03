"""Resample an OpenDuck motion NPZ and rebuild its full-body kinematics."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.assets.robots.open_duck_mini_v2.open_duck_constants import OPEN_DUCK_XML


def _linear_resample(
  values: np.ndarray,
  source_times: np.ndarray,
  target_times: np.ndarray,
) -> np.ndarray:
  flattened = values.reshape(values.shape[0], -1)
  result = np.empty((target_times.size, flattened.shape[1]), dtype=np.float64)
  for index in range(flattened.shape[1]):
    result[:, index] = np.interp(target_times, source_times, flattened[:, index])
  return result.reshape((target_times.size, *values.shape[1:]))


def _quaternion_resample(
  quaternions_wxyz: np.ndarray,
  source_times: np.ndarray,
  target_times: np.ndarray,
) -> np.ndarray:
  rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]])
  interpolated_xyzw = Slerp(source_times, rotations)(target_times).as_quat()
  return interpolated_xyzw[:, [3, 0, 1, 2]]


def _angular_velocity_world(
  quaternions_wxyz: np.ndarray,
  dt: float,
) -> np.ndarray:
  rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]])
  angular_velocity = np.zeros((quaternions_wxyz.shape[0], 3), dtype=np.float64)
  relative = rotations[2:] * rotations[:-2].inv()
  angular_velocity[1:-1] = relative.as_rotvec() / (2.0 * dt)
  angular_velocity[0] = angular_velocity[1]
  angular_velocity[-1] = angular_velocity[-2]
  return angular_velocity


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
  return [
    mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"
    for index in range(count)
  ]


def resample_motion(
  input_path: Path,
  output_path: Path,
  output_fps: float = 50.0,
) -> None:
  source = np.load(input_path)
  required_keys = {
    "fps",
    "joint_pos",
    "body_pos_w",
    "body_quat_w",
  }
  missing = required_keys.difference(source.files)
  if missing:
    raise KeyError(f"Motion is missing required arrays: {sorted(missing)}")

  source_fps = float(source["fps"])
  source_joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
  source_root_pos = np.asarray(source["body_pos_w"][:, 0], dtype=np.float64)
  source_root_quat = np.asarray(source["body_quat_w"][:, 0], dtype=np.float64)

  source_times = np.arange(source_joint_pos.shape[0], dtype=np.float64) / source_fps
  source_duration = source_times[-1]
  target_dt = 1.0 / output_fps
  target_times = np.arange(
    0.0,
    source_duration + 0.5 * target_dt,
    target_dt,
    dtype=np.float64,
  )
  target_times = target_times[target_times <= source_duration]

  joint_pos = _linear_resample(source_joint_pos, source_times, target_times)
  root_pos = _linear_resample(source_root_pos, source_times, target_times)
  root_quat = _quaternion_resample(source_root_quat, source_times, target_times)
  joint_vel = np.gradient(joint_pos, target_dt, axis=0)
  root_lin_vel = np.gradient(root_pos, target_dt, axis=0)
  root_ang_vel = _angular_velocity_world(root_quat, target_dt)

  model = mujoco.MjModel.from_xml_path(str(OPEN_DUCK_XML))
  data = mujoco.MjData(model)
  if model.nq != 7 + joint_pos.shape[1]:
    raise ValueError(
      f"Motion has {joint_pos.shape[1]} joints but model expects {model.nq - 7}"
    )

  frame_count = target_times.size
  body_count = model.nbody - 1
  body_pos_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_quat_w = np.empty((frame_count, body_count, 4), dtype=np.float32)
  body_lin_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_ang_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)

  for frame in range(frame_count):
    data.qpos[:3] = root_pos[frame]
    data.qpos[3:7] = root_quat[frame]
    data.qpos[7:] = joint_pos[frame]
    data.qvel[:3] = root_lin_vel[frame]
    data.qvel[3:6] = root_ang_vel[frame]
    data.qvel[6:] = joint_vel[frame]
    mujoco.mj_forward(model, data)

    body_pos_w[frame] = data.xpos[1:]
    body_quat_w[frame] = data.xquat[1:]
    root_subtree_com = data.subtree_com[1]
    for body_index, body_id in enumerate(range(1, model.nbody)):
      angular = data.cvel[body_id, :3]
      linear = data.cvel[body_id, 3:]
      offset = root_subtree_com - data.xpos[body_id]
      body_lin_vel_w[frame, body_index] = linear - np.cross(angular, offset)
      body_ang_vel_w[frame, body_index] = angular

  joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)[1:]
  body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)[1:]

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    fps=np.float64(output_fps),
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel.astype(np.float32),
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    joint_names=np.asarray(joint_names),
    body_names=np.asarray(body_names),
  )

  generated = np.load(output_path)
  arrays = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  for key in arrays:
    if not np.isfinite(generated[key]).all():
      raise ValueError(f"Generated array {key} contains non-finite values")
  np.testing.assert_allclose(
    np.linalg.norm(generated["body_quat_w"], axis=-1),
    1.0,
    atol=1.0e-5,
  )

  print(f"Input:  {input_path} ({source_joint_pos.shape[0]} frames @ {source_fps:.6g} Hz)")
  print(f"Output: {output_path} ({frame_count} frames @ {output_fps:g} Hz)")
  print(f"Duration: {target_times[-1]:.3f} s")
  print(f"Joints: {len(joint_names)}, bodies: {len(body_names)}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--output-fps", type=float, default=50.0)
  args = parser.parse_args()
  resample_motion(args.input, args.output, args.output_fps)


if __name__ == "__main__":
  main()
