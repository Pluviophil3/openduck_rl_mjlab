from __future__ import annotations

import unittest

from mjlab.actuator import DelayedActuatorCfg

import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

from src.tasks.tracking.config.open_duck.randomization import (
  RANDOMIZATION_PROFILE_NAMES,
  apply_open_duck_randomization,
  profile_as_dict,
)


TASK_ID = "OpenDuck-Tracking-No-State-Estimation"


class OpenDuckRandomizationTest(unittest.TestCase):
  def setUp(self) -> None:
    self.cfg = load_env_cfg(TASK_ID)

  def test_nominal_is_fully_deterministic(self) -> None:
    apply_open_duck_randomization(self.cfg, "nominal")

    self.assertEqual(self.cfg.events, {})
    self.assertFalse(self.cfg.observations["actor"].enable_corruption)
    motion = self.cfg.commands["motion"]
    self.assertEqual(motion.pose_range, {})
    self.assertEqual(motion.velocity_range, {})
    self.assertEqual(motion.joint_position_range, (0.0, 0.0))
    self.assertTrue(
      all(
        term.delay_max_lag == 0
        for term in self.cfg.observations["actor"].terms.values()
      )
    )

  def test_baseline_reproduces_sway_t2_training_profile(self) -> None:
    apply_open_duck_randomization(self.cfg, "baseline")

    self.assertEqual(
      set(self.cfg.events),
      {"push_robot", "base_com", "encoder_bias", "foot_friction"},
    )
    self.assertTrue(self.cfg.observations["actor"].enable_corruption)
    self.assertEqual(
      self.cfg.events["foot_friction"].params["ranges"], (0.3, 1.2)
    )
    self.assertEqual(
      self.cfg.events["encoder_bias"].params["bias_range"], (-0.01, 0.01)
    )

  def test_full_enables_all_supported_randomizations(self) -> None:
    apply_open_duck_randomization(self.cfg, "full")

    self.assertEqual(
      set(self.cfg.events),
      {
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
      },
    )
    actuators = self.cfg.scene.entities["robot"].articulation.actuators
    self.assertTrue(all(isinstance(a, DelayedActuatorCfg) for a in actuators))
    self.assertTrue(all(a.delay_max_lag == 8 for a in actuators))
    actor_terms = self.cfg.observations["actor"].terms
    for name in ("base_ang_vel", "joint_pos", "joint_vel"):
      self.assertEqual(actor_terms[name].delay_max_lag, 2)

  def test_profiles_can_be_switched_without_nested_delay_wrappers(self) -> None:
    apply_open_duck_randomization(self.cfg, "full")
    apply_open_duck_randomization(self.cfg, "light")
    apply_open_duck_randomization(self.cfg, "nominal")

    actuators = self.cfg.scene.entities["robot"].articulation.actuators
    self.assertTrue(all(not isinstance(a, DelayedActuatorCfg) for a in actuators))

  def test_every_profile_has_serializable_metadata(self) -> None:
    for name in RANDOMIZATION_PROFILE_NAMES:
      metadata = profile_as_dict(name)
      self.assertEqual(metadata["name"], name)


if __name__ == "__main__":
  unittest.main()
