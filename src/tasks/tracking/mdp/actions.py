from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class OpenDuckJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with runtime-style safety processing."""

  action_clip: float | None = 20.0
  # Rate limits and low-pass filters should be enabled deliberately for
  # sim-to-real fine-tuning; the reference motions can require much larger
  # per-step joint changes than the real-time safety defaults.
  max_target_step: float | None = None
  safety_clip: bool = True
  joint_limit_margin: float = 0.02
  cutoff_frequency: float | None = None

  def build(self, env):
    return OpenDuckJointPositionAction(self, env)


class OpenDuckJointPositionAction(JointPositionAction):
  cfg: OpenDuckJointPositionActionCfg

  def __init__(self, cfg: OpenDuckJointPositionActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    self._prev_clipped_actions = torch.zeros_like(self._raw_actions)
    self._prev_prev_clipped_actions = torch.zeros_like(self._raw_actions)
    self._last_target = self._current_joint_pos().clone()
    self._target_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._filter_alpha = self._compute_filter_alpha(cfg.cutoff_frequency, env.step_dt)

  @staticmethod
  def _compute_filter_alpha(cutoff_frequency: float | None, step_dt: float) -> float | None:
    if cutoff_frequency is None:
      return None
    if cutoff_frequency <= 0.0:
      raise ValueError("cutoff_frequency must be positive when set")
    cutoff_period = 1.0 / float(cutoff_frequency)
    return cutoff_period / (float(step_dt) + cutoff_period)

  @property
  def clipped_action_history(self) -> torch.Tensor:
    return torch.cat(
      [
        self._raw_actions,
        self._prev_clipped_actions,
        self._prev_prev_clipped_actions,
      ],
      dim=-1,
    )

  def _current_joint_pos(self) -> torch.Tensor:
    return self._entity.data.joint_pos[:, self._target_ids]

  def _expanded_offset(self) -> torch.Tensor | float:
    if isinstance(self._offset, torch.Tensor):
      return self._offset
    return float(self._offset)

  def _previous_target(self) -> torch.Tensor:
    current = self._current_joint_pos()
    return torch.where(
      self._target_initialized.unsqueeze(-1), self._last_target, current
    )

  def _apply_target_rate_limit(self, target: torch.Tensor) -> torch.Tensor:
    if self.cfg.max_target_step is None:
      return target
    if self.cfg.max_target_step <= 0.0:
      raise ValueError("max_target_step must be positive when set")
    previous = self._previous_target()
    delta = torch.clamp(
      target - previous,
      min=-self.cfg.max_target_step,
      max=self.cfg.max_target_step,
    )
    return previous + delta

  def _apply_joint_limit_clip(self, target: torch.Tensor) -> torch.Tensor:
    if not self.cfg.safety_clip:
      return target
    limits = self._entity.data.joint_pos_limits[:, self._target_ids]
    lower = limits[..., 0] + self.cfg.joint_limit_margin
    upper = limits[..., 1] - self.cfg.joint_limit_margin
    center = 0.5 * (limits[..., 0] + limits[..., 1])
    lower = torch.minimum(lower, center)
    upper = torch.maximum(upper, center)
    return torch.clamp(target, min=lower, max=upper)

  def _apply_low_pass_filter(self, target: torch.Tensor) -> torch.Tensor:
    if self._filter_alpha is None:
      return target
    previous = self._previous_target()
    return self._filter_alpha * previous + (1.0 - self._filter_alpha) * target

  def process_actions(self, actions: torch.Tensor):
    action = actions.to(self.device)
    if self.cfg.action_clip is not None:
      if self.cfg.action_clip <= 0.0:
        raise ValueError("action_clip must be positive when set")
      action = torch.clamp(action, -self.cfg.action_clip, self.cfg.action_clip)

    self._prev_prev_clipped_actions[:] = self._prev_clipped_actions
    self._prev_clipped_actions[:] = self._raw_actions
    self._raw_actions[:] = action

    target = self._raw_actions * self._scale + self._expanded_offset()
    target = self._apply_target_rate_limit(target)
    target = self._apply_joint_limit_clip(target)
    target = self._apply_low_pass_filter(target)

    self._processed_actions = target
    self._last_target[:] = target
    self._target_initialized[:] = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    self._prev_clipped_actions[env_ids] = 0.0
    self._prev_prev_clipped_actions[env_ids] = 0.0
    self._last_target[env_ids] = self._current_joint_pos()[env_ids]
    self._target_initialized[env_ids] = False
