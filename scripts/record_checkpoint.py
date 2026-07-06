"""Record a headless motion-tracking video for one checkpoint."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class RecordConfig:
  checkpoint_file: str
  motion_file: str
  video_folder: str
  num_envs: int = 1
  video_length: int = 500
  device: str | None = None
  video_height: int = 480
  video_width: int = 640


def run_record(task_id: str, cfg: RecordConfig) -> None:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  motion_path = Path(cfg.motion_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
  if not motion_path.is_file():
    raise FileNotFoundError(f"Motion not found: {motion_path}")

  video_folder = Path(cfg.video_folder).expanduser().resolve()
  video_folder.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError(f"Task {task_id!r} is not a motion-tracking task")
  motion_cfg.motion_file = str(motion_path)

  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=video_folder,
    step_trigger=lambda step: step == 0,
    video_length=cfg.video_length,
    name_prefix=checkpoint_path.stem,
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  print(
    f"[record] Recording {cfg.video_length} steps from {checkpoint_path.name} "
    f"with {motion_path.name}"
  )
  observations, _ = env.reset()
  for _ in range(cfg.video_length + 1):
    with torch.no_grad():
      actions = policy(observations)
    observations, _, _, _ = env.step(actions)

  env.close()
  print(f"[record] Video saved under: {video_folder}")


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
  )
  cfg = tyro.cli(RecordConfig, args=remaining_args)
  run_record(task_id, cfg)


if __name__ == "__main__":
  main()
