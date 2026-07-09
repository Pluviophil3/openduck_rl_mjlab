"""Headless, reproducible evaluation for OpenDuck tracking checkpoints."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Both this repository and sibling workspaces expose a top-level ``src`` package.
# Pin direct script execution to this checkout before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
  sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.tracking.config.open_duck.randomization import (
  RandomizationProfileName,
  apply_open_duck_randomization,
  profile_as_dict,
)
from src.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from src.tasks.tracking.mdp.metrics import compute_root_relative_mpkpe


@dataclass(frozen=True)
class EvaluationConfig:
  checkpoint_file: str = "models/Sway_t2/model_2500.pt"
  motion_file: str = "src/assets/motions/open_duck/A2_-_Sway_t2_stageii.npz"
  output_dir: str = "models/Sway_t2/nominal"
  task_id: str = "OpenDuck-Tracking-No-State-Estimation"
  profile: RandomizationProfileName = "nominal"
  num_envs: int = 1
  num_steps: int | None = None
  seed: int = 42
  device: str = "cuda:0"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_revision() -> str:
  result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _summary(values: list[float]) -> dict[str, float]:
  array = np.asarray(values, dtype=np.float64)
  return {
    "mean": float(np.mean(array)),
    "std": float(np.std(array)),
    "p95": float(np.percentile(array, 95)),
    "p99": float(np.percentile(array, 99)),
    "max": float(np.max(array)),
  }


def run_evaluation(cfg: EvaluationConfig) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)

  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  motion_path = Path(cfg.motion_file).expanduser().resolve()
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
  if not motion_path.is_file():
    raise FileNotFoundError(f"Motion not found: {motion_path}")
  output_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)
  apply_open_duck_randomization(env_cfg, cfg.profile)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs

  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError(f"{cfg.task_id!r} is not a motion-tracking task")
  motion_cfg.motion_file = str(motion_path)
  motion_cfg.sampling_mode = "start"

  motion_data = np.load(motion_path)
  num_steps = cfg.num_steps or int(motion_data["joint_pos"].shape[0])

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=cfg.device,
  )
  policy = runner.get_inference_policy(device=cfg.device)

  command = raw_env.command_manager.get_term("motion")
  if not isinstance(command, MotionCommand):
    raise TypeError("Expected MotionCommand")

  observations, _ = env.reset()
  metric_series: dict[str, list[float]] = {}
  reward_series: list[float] = []
  termination_counts = {
    name: 0 for name in raw_env.termination_manager.active_terms
  }
  termination_steps: dict[str, list[int]] = {
    name: [] for name in raw_env.termination_manager.active_terms
  }
  ever_terminated = torch.zeros(
    cfg.num_envs, dtype=torch.bool, device=cfg.device
  )
  trace: dict[str, list[np.ndarray]] = {
    "actor_observation": [],
    "action": [],
    "joint_pos": [],
    "joint_vel": [],
    "joint_pos_target": [],
    "reference_joint_pos": [],
    "reference_joint_vel": [],
    "reward": [],
  }

  for step in range(num_steps):
    actor_observation = observations["actor"]
    with torch.inference_mode():
      actions = policy(observations)
    observations, rewards, _, _ = env.step(actions)

    for name, value in command.metrics.items():
      metric_series.setdefault(name, []).append(
        float(value.float().mean().item())
      )
    metric_series.setdefault("root_relative_mpkpe", []).append(
      float(compute_root_relative_mpkpe(command).mean().item())
    )
    reward_series.append(float(rewards.mean().item()))

    terminated = raw_env.termination_manager.terminated.clone()
    ever_terminated |= terminated
    for name in raw_env.termination_manager.active_terms:
      count = int(raw_env.termination_manager.get_term(name).sum().item())
      termination_counts[name] += count
      if count:
        termination_steps[name].append(step)

    robot = raw_env.scene["robot"]
    trace["actor_observation"].append(
      actor_observation[0].detach().cpu().numpy().copy()
    )
    trace["action"].append(actions[0].detach().cpu().numpy().copy())
    trace["joint_pos"].append(
      robot.data.joint_pos[0].detach().cpu().numpy().copy()
    )
    trace["joint_vel"].append(
      robot.data.joint_vel[0].detach().cpu().numpy().copy()
    )
    trace["joint_pos_target"].append(
      robot.data.joint_pos_target[0].detach().cpu().numpy().copy()
    )
    trace["reference_joint_pos"].append(
      command.joint_pos[0].detach().cpu().numpy().copy()
    )
    trace["reference_joint_vel"].append(
      command.joint_vel[0].detach().cpu().numpy().copy()
    )
    trace["reward"].append(np.asarray(rewards[0].item(), dtype=np.float32))

  action_array = np.stack(trace["action"])
  action_term = raw_env.action_manager.get_term("joint_pos")
  action_target_names = list(getattr(action_term, "target_names"))
  summary = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_revision": _git_revision(),
    "configuration": asdict(cfg),
    "resolved_num_steps": num_steps,
    "control_frequency_hz": 1.0 / raw_env.step_dt,
    "checkpoint": {
      "path": str(checkpoint_path),
      "sha256": _sha256(checkpoint_path),
    },
    "motion": {
      "path": str(motion_path),
      "sha256": _sha256(motion_path),
      "frames": int(motion_data["joint_pos"].shape[0]),
      "fps": float(motion_data["fps"]),
    },
    "randomization_profile": profile_as_dict(cfg.profile),
    "metrics": {
      name: _summary(values) for name, values in metric_series.items()
    },
    "reward": _summary(reward_series),
    "action": {
      "target_names": action_target_names,
      "abs_mean": float(np.mean(np.abs(action_array))),
      "abs_p95": float(np.percentile(np.abs(action_array), 95)),
      "abs_p99": float(np.percentile(np.abs(action_array), 99)),
      "abs_max": float(np.max(np.abs(action_array))),
      "fraction_abs_gt_1": float(np.mean(np.abs(action_array) > 1.0)),
      "per_target_abs_max": {
        name: float(value)
        for name, value in zip(
          action_target_names,
          np.max(np.abs(action_array), axis=0),
          strict=True,
        )
      },
    },
    "termination_counts": termination_counts,
    "termination_steps": termination_steps,
    "environments_without_failure_fraction": float(
      (~ever_terminated).float().mean().item()
    ),
  }

  np.savez_compressed(
    output_dir / "trace.npz",
    **{name: np.stack(values) for name, values in trace.items()},
  )
  (output_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
  )
  env.close()

  print(json.dumps(summary, indent=2, ensure_ascii=False))
  print(f"Saved summary and trace under {output_dir}")


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  run_evaluation(tyro.cli(EvaluationConfig))


if __name__ == "__main__":
  main()
