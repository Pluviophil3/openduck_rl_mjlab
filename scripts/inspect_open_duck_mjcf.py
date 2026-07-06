"""Export compiled OpenDuck MJCF physical parameters as JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import tyro


@dataclass(frozen=True)
class InspectConfig:
  mjcf: str = (
    "src/assets/robots/open_duck_mini_v2/xmls/open_duck_mini_v2.xml"
  )
  output: str = "doc/open_duck_mjcf_report.json"


def _name(model: mujoco.MjModel, object_type: int, index: int) -> str:
  return mujoco.mj_id2name(model, object_type, index) or f"<unnamed-{index}>"


def inspect_model(cfg: InspectConfig) -> None:
  mjcf_path = Path(cfg.mjcf).expanduser().resolve()
  output_path = Path(cfg.output).expanduser().resolve()
  model = mujoco.MjModel.from_xml_path(str(mjcf_path))

  body_records = [
    {
      "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, index),
      "mass_kg": float(model.body_mass[index]),
      "inertial_position_m": model.body_ipos[index].tolist(),
      "principal_inertia_kg_m2": model.body_inertia[index].tolist(),
    }
    for index in range(1, model.nbody)
  ]
  joint_records = []
  for index in range(1, model.njnt):  # Skip the floating-base free joint.
    dof_index = model.jnt_dofadr[index]
    joint_records.append(
      {
        "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
        "range_rad": model.jnt_range[index].tolist(),
        "damping": float(model.dof_damping[dof_index]),
        "frictionloss": float(model.dof_frictionloss[dof_index]),
        "armature": float(model.dof_armature[dof_index]),
      }
    )
  actuator_records = [
    {
      "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index),
      "control_range_rad": model.actuator_ctrlrange[index].tolist(),
      "force_range": model.actuator_forcerange[index].tolist(),
      "kp": float(model.actuator_gainprm[index, 0]),
      "kv": float(-model.actuator_biasprm[index, 2]),
    }
    for index in range(model.nu)
  ]
  foot_contact_records = [
    {
      "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, index),
      "friction": model.geom_friction[index].tolist(),
      "solref": model.geom_solref[index].tolist(),
      "solimp": model.geom_solimp[index].tolist(),
    }
    for index in range(model.ngeom)
    if "foot_bottom_tpu"
    in _name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
  ]

  report = {
    "mjcf": str(mjcf_path),
    "mujoco_version": mujoco.__version__,
    "compiled_counts": {
      "bodies_excluding_world": model.nbody - 1,
      "joints_including_free_joint": model.njnt,
      "actuators_in_xml": model.nu,
      "position_coordinates": model.nq,
      "velocity_coordinates": model.nv,
    },
    "total_mass_kg": float(model.body_mass.sum()),
    "xml_timestep_s": float(model.opt.timestep),
    "gravity_m_s2": model.opt.gravity.tolist(),
    "bodies": body_records,
    "joints": joint_records,
    "actuators": actuator_records,
    "foot_contacts": foot_contact_records,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  print(f"Saved report to {output_path}")


if __name__ == "__main__":
  inspect_model(tyro.cli(InspectConfig))
