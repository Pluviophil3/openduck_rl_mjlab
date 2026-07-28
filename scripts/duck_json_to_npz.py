#!/usr/bin/env python3
"""Convert OpenDuck generator JSON recordings to unitree_rl_mjlab NPZ files.

The default target model is open_duck_mini_v2_real.xml, which contains the
passive *_backlash joints used by the tracking environment. Generator JSON files
store only the main joints, so backlash columns are emitted as zeros.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
  PROJECT_ROOT
  / "src"
  / "assets"
  / "robots"
  / "open_duck_mini_v2"
  / "xmls"
  / "open_duck_mini_v2_real.xml"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "assets" / "motions" / "open_duck"


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
  return [
    mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"
    for index in range(count)
  ]


def _frame_layout(recording: dict) -> tuple[dict[str, int], dict[str, int]]:
  if not recording.get("Frame_offset") or not recording.get("Frame_size"):
    raise ValueError("Recording is missing Frame_offset/Frame_size metadata")
  return recording["Frame_offset"][0], recording["Frame_size"][0]


def _slice(
  frames: np.ndarray,
  offsets: dict[str, int],
  sizes: dict[str, int],
  name: str,
) -> np.ndarray:
  start = offsets[name]
  return frames[:, start : start + sizes[name]]


def _joint_limits(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
  limits: dict[str, tuple[float, float]] = {}
  for joint_id in range(1, model.njnt):
    if not model.jnt_limited[joint_id]:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is not None:
      limits[name] = tuple(float(v) for v in model.jnt_range[joint_id])
  return limits


def _build_joint_pos(
  recording: dict,
  model: mujoco.MjModel,
  json_joint_pos: np.ndarray,
  clip_to_limits: bool,
) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
  source_joint_names = [str(name) for name in recording["Joints"]]
  source_index_by_name = {
    name: index for index, name in enumerate(source_joint_names)
  }
  target_joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)[1:]
  limits = _joint_limits(model)
  joint_pos = np.zeros((json_joint_pos.shape[0], len(target_joint_names)), dtype=np.float64)
  clip_report: dict[str, dict[str, float]] = {}

  for target_index, target_name in enumerate(target_joint_names):
    if target_name.endswith("_backlash"):
      continue
    source_index = source_index_by_name.get(target_name)
    if source_index is None:
      raise ValueError(f"Recording is missing required joint: {target_name}")

    values = json_joint_pos[:, source_index].astype(np.float64)
    if clip_to_limits and target_name in limits:
      lower, upper = limits[target_name]
      clipped = np.clip(values, lower, upper)
      if np.any(clipped != values):
        clip_report[target_name] = {
          "source_min": float(values.min()),
          "source_max": float(values.max()),
          "limit_min": lower,
          "limit_max": upper,
          "clipped_frames": int(np.count_nonzero(clipped != values)),
          "max_abs_delta": float(np.max(np.abs(clipped - values))),
        }
      values = clipped
    joint_pos[:, target_index] = values

  return joint_pos, target_joint_names, clip_report


def _angular_velocity_world(quaternions_wxyz: np.ndarray, dt: float) -> np.ndarray:
  angular_velocity = np.zeros((*quaternions_wxyz.shape[:2], 3), dtype=np.float64)
  for body_index in range(quaternions_wxyz.shape[1]):
    rotations = Rotation.from_quat(quaternions_wxyz[:, body_index, [1, 2, 3, 0]])
    relative = rotations[2:] * rotations[:-2].inv()
    angular_velocity[1:-1, body_index] = relative.as_rotvec() / (2.0 * dt)
  angular_velocity[0] = angular_velocity[1]
  angular_velocity[-1] = angular_velocity[-2]
  return angular_velocity


def export_recording(
  input_path: Path,
  output_path: Path,
  xml_path: Path,
  clip_to_limits: bool,
) -> None:
  with input_path.open("r") as file:
    recording = json.load(file)

  fps = float(recording["FPS"])
  dt = 1.0 / fps
  frames = np.asarray(recording["Frames"], dtype=np.float64)
  offsets, sizes = _frame_layout(recording)

  root_pos = _slice(frames, offsets, sizes, "root_pos")
  root_quat_xyzw = _slice(frames, offsets, sizes, "root_quat")
  root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
  json_joint_pos = _slice(frames, offsets, sizes, "joints_pos")

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  joint_pos, joint_names, clip_report = _build_joint_pos(
    recording, model, json_joint_pos, clip_to_limits
  )
  joint_vel = np.gradient(joint_pos, dt, axis=0)

  if model.nq != 7 + joint_pos.shape[1]:
    raise ValueError(
      f"Model expects {model.nq - 7} joints, built {joint_pos.shape[1]}"
    )

  frame_count = frames.shape[0]
  body_count = model.nbody - 1
  body_pos_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_quat_w = np.empty((frame_count, body_count, 4), dtype=np.float32)
  body_lin_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_ang_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)

  root_lin_vel = _slice(frames, offsets, sizes, "world_linear_vel")
  root_ang_vel = _slice(frames, offsets, sizes, "world_angular_vel")

  for frame in range(frame_count):
    data.qpos[:3] = root_pos[frame]
    data.qpos[3:7] = root_quat_wxyz[frame]
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

  # MuJoCo cvel can be brittle for the first frame if qvel is discontinuous.
  # Recompute angular velocity from body quaternions for stable edge samples.
  body_ang_vel_w = _angular_velocity_world(body_quat_w.astype(np.float64), dt).astype(np.float32)
  body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)[1:]

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    fps=np.float64(fps),
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
  for key in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    if not np.isfinite(generated[key]).all():
      raise ValueError(f"Generated array {key} contains non-finite values")
  np.testing.assert_allclose(
    np.linalg.norm(generated["body_quat_w"], axis=-1),
    1.0,
    atol=1.0e-5,
  )

  backlash_count = sum(name.endswith("_backlash") for name in joint_names)
  print(f"Input:  {input_path}")
  print(f"XML:    {xml_path}")
  print(f"Output: {output_path}")
  print(f"Frames: {frame_count} @ {fps:g} Hz")
  print(f"Joints: {len(joint_names)} ({backlash_count} backlash)")
  print(f"Bodies: {len(body_names)}")
  if clip_report:
    print("Clipped joints:")
    for name, report in clip_report.items():
      print(
        f"  {name}: {report['clipped_frames']} frames, "
        f"source=[{report['source_min']:.6f}, {report['source_max']:.6f}], "
        f"limit=[{report['limit_min']:.6f}, {report['limit_max']:.6f}], "
        f"max_delta={report['max_abs_delta']:.6f}"
      )


def _iter_json_files(input_dir: Path) -> list[Path]:
  return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert OpenDuck generator JSON to unitree_rl_mjlab NPZ"
  )
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--input", type=Path, help="Single generator JSON recording")
  group.add_argument("--input-dir", type=Path, help="Directory of generator JSON files")
  parser.add_argument("--output", type=Path, help="Output NPZ path for --input")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Output directory for --input-dir or default single-file output",
  )
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument(
    "--no-clip",
    action="store_true",
    help="Do not clip main joints to the MuJoCo XML joint ranges",
  )
  args = parser.parse_args()

  if args.input is not None:
    output = args.output or args.output_dir / f"{args.input.stem}.npz"
    export_recording(args.input, output, args.xml, not args.no_clip)
    return

  assert args.input_dir is not None
  json_files = _iter_json_files(args.input_dir)
  if not json_files:
    raise SystemExit(f"No JSON files found in {args.input_dir}")

  print(f"Converting {len(json_files)} files into {args.output_dir}")
  for json_path in json_files:
    output_path = args.output_dir / f"{json_path.stem}.npz"
    export_recording(json_path, output_path, args.xml, not args.no_clip)


if __name__ == "__main__":
  main()
