#!/usr/bin/env python3
"""Steady-state evaluation for trained Silver Badger policies.

The script has two modes:

1. ``collect`` loads one trained policy through the standard legged_gym task
   registry, evaluates a grid of fixed gaits and forward-speed commands, and
   writes episode-level metrics plus optional compressed time series.
2. ``compare`` combines the results of several variants and creates aggregate
   CSV tables and plots.

Run ``collect`` once for each policy variant using the same evaluation options.
The variant label is inferred from the loaded Silver Badger ablation config, but
it can be overridden with ``--variant``.
The remaining command-line arguments (for example ``--task``, ``--load_run``,
``--checkpoint``, and ``--headless``) are forwarded to legged_gym's get_args().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


DEFAULT_GAITS = [
    "TROT",
    "WALK",
    "PACE",
    "BOUND",
    "PRONK",
    "CANTER",
    "TRANSVERSE_GALLOP",
    "ROTARY_GALLOP",
    "AMBLE",
]

DEFAULT_SPEEDS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

SUMMARY_FIELDS = [
    "variant",
    "task",
    "spine_control_mode",
    "spine_phase_mode",
    "spine_phase_source",
    "spine_phase_offset_rad",
    "num_actions",
    "num_observations",
    "num_cpgs",
    "cpg_parameter_mode",
    "robot_height_m",
    "ground_clearance_m",
    "ground_penetration_m",
    "mean_offset_x_m",
    "offset_x_control",
    "gait",
    "command_mps",
    "episode",
    "fell",
    "valid_steps",
    "measurement_time_s",
    "mean_velocity_mps",
    "velocity_bias_mps",
    "velocity_mae_mps",
    "velocity_rmse_mps",
    "phase_error_rad",
    "phase_error_deg",
    "roll_rms_rad",
    "pitch_rms_rad",
    "mean_mechanical_power_w",
    "total_mechanical_energy_j",
    "leg_mechanical_energy_j",
    "spine_mechanical_energy_j",
    "distance_m",
    "cost_of_transport",
    "leg_torque_saturation_fraction",
    "requested_leg_torque_saturation_fraction",
    "peak_foot_contact_force_n",
    "spine_angle_rms_rad",
    "spine_angle_range_rad",
    "spine_leg_phase_locking_value",
    "spine_leg_mean_phase_rad",
    "spine_target_phase_error_rad",
    "robot_mass_kg",
]

AGGREGATE_METRICS = [
    "mean_velocity_mps",
    "velocity_bias_mps",
    "velocity_mae_mps",
    "velocity_rmse_mps",
    "phase_error_rad",
    "phase_error_deg",
    "roll_rms_rad",
    "pitch_rms_rad",
    "mean_mechanical_power_w",
    "total_mechanical_energy_j",
    "leg_mechanical_energy_j",
    "spine_mechanical_energy_j",
    "distance_m",
    "cost_of_transport",
    "leg_torque_saturation_fraction",
    "requested_leg_torque_saturation_fraction",
    "peak_foot_contact_force_n",
    "spine_angle_rms_rad",
    "spine_angle_range_rad",
    "spine_leg_phase_locking_value",
    "spine_leg_mean_phase_rad",
    "spine_target_phase_error_rad",
    "fell",
]


def parse_evaluation_args(argv: Sequence[str] | None = None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse evaluator options and return unconsumed legged_gym arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-mode", choices=("collect", "compare"), default="collect")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("steady_state_results"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--gaits", nargs="+", default=DEFAULT_GAITS)
    parser.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--measure-s", type=float, default=10.0)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--contact-threshold-n", type=float, default=1.0)
    parser.add_argument("--robot-mass-kg", type=float, default=None)
    parser.add_argument(
        "--cpg-parameter-mode",
        choices=("nominal", "randomized"),
        default="nominal",
        help="Use training-range midpoints or deterministic episode-level samples.",
    )
    parser.add_argument("--robot-height-m", type=float, default=None)
    parser.add_argument("--ground-clearance-m", type=float, default=None)
    parser.add_argument("--ground-penetration-m", type=float, default=None)
    parser.add_argument("--offset-x-m", type=float, default=None)
    parser.add_argument("--save-timeseries", action="store_true")
    parser.add_argument("--keep-domain-rand", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--steady-help", action="store_true")
    args, remaining = parser.parse_known_args(argv)

    if args.steady_help:
        print(__doc__)
        print("Evaluator options:\n")
        parser.print_help()
        print("\nAll remaining options are passed to legged_gym.utils.get_args().")
        raise SystemExit(0)

    return args, remaining


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def set_if_present(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def configure_evaluation_environment(env_cfg: Any, eval_args: argparse.Namespace) -> None:
    """Make evaluation deterministic and keep commands/gaits fixed."""
    env_cfg.env.num_envs = eval_args.episodes
    set_if_present(env_cfg.env, "play", True)
    env_cfg.env.episode_length_s = max(
        float(env_cfg.env.episode_length_s),
        eval_args.warmup_s + eval_args.measure_s + 2.0,
    )

    set_if_present(env_cfg.commands, "curriculum", False)
    set_if_present(env_cfg.commands, "resample_gait_style", False)
    set_if_present(env_cfg.commands, "resampling_time", 1.0e9)
    set_if_present(env_cfg.commands, "gait_resampling_time", 1.0e9)

    first_speed = float(eval_args.speeds[0])
    if hasattr(env_cfg.commands, "ranges"):
        set_if_present(env_cfg.commands.ranges, "lin_vel_x", [first_speed, first_speed])
        set_if_present(env_cfg.commands.ranges, "lin_vel_y", [0.0, 0.0])
        set_if_present(env_cfg.commands.ranges, "ang_vel_yaw", [0.0, 0.0])

    set_if_present(env_cfg.noise, "add_noise", False)

    if not eval_args.keep_domain_rand:
        for name in (
            "randomize_friction",
            "randomize_base_mass",
            "push_robots",
            "randomize_lag_timesteps",
            "randomize_PD",
            "latency",
        ):
            set_if_present(env_cfg.domain_rand, name, False)

        set_if_present(env_cfg.terrain, "mesh_type", "plane")
        set_if_present(env_cfg.terrain, "curriculum", False)
        set_if_present(env_cfg.terrain, "measure_heights", False)


def get_robot_mass(env: Any, override_kg: float | None) -> float:
    if override_kg is not None:
        return float(override_kg)
    try:
        props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
        return float(sum(float(prop.mass) for prop in props))
    except Exception:
        return math.nan


def wrap_angle_torch(angle: Any) -> Any:
    import torch

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def set_fixed_gait(env: Any, gait: str) -> int:
    gait = gait.upper()
    names = [name.upper() for name in env._cpg.gait_names]
    if gait not in names:
        raise ValueError(f"Gait {gait!r} is unavailable. Available gaits: {names}")
    index = names.index(gait)
    matrix = env._cpg.available_gaits[index]
    env._cpg.PHI_batch[:] = matrix
    env._cpg.current_gait_indices[:] = index
    if hasattr(env._cpg, "previous_gait_indices"):
        env._cpg.previous_gait_indices[:] = index
    return index


def set_fixed_command(env: Any, speed: float) -> None:
    env.commands.zero_()
    env.commands[:, 0] = float(speed)


def _parameter_values(
    value_range: Sequence[float],
    override: float | None,
    mode: str,
    num_envs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if override is not None:
        return np.full(num_envs, float(override), dtype=np.float32)
    low, high = map(float, value_range)
    if mode == "nominal":
        return np.full(num_envs, 0.5 * (low + high), dtype=np.float32)
    return rng.uniform(low, high, size=num_envs).astype(np.float32)


def set_cpg_episode_parameters(
    env: Any,
    eval_args: argparse.Namespace,
    rng: np.random.Generator,
    torch: Any,
) -> Dict[str, Any]:
    """Set nominal or reproducibly randomized CPG trajectory geometry."""
    cpg = env._cpg
    num_envs = env.num_envs
    height = _parameter_values(
        cpg.robot_height_range,
        eval_args.robot_height_m,
        eval_args.cpg_parameter_mode,
        num_envs,
        rng,
    )
    clearance = _parameter_values(
        cpg.ground_clearance_range,
        eval_args.ground_clearance_m,
        eval_args.cpg_parameter_mode,
        num_envs,
        rng,
    )
    penetration = _parameter_values(
        cpg.ground_penetration_range,
        eval_args.ground_penetration_m,
        eval_args.cpg_parameter_mode,
        num_envs,
        rng,
    )

    cpg._robot_height[:] = torch.as_tensor(height, device=env.device)
    cpg._ground_clearance[:] = torch.as_tensor(clearance, device=env.device)
    cpg._ground_penetration[:] = torch.as_tensor(penetration, device=env.device)

    offset_control = "absent"
    if hasattr(cpg, "_offset_x"):
        if bool(getattr(cpg, "_offset_x_from_actions", False)):
            if eval_args.offset_x_m is not None:
                raise ValueError(
                    "--offset-x-m cannot be imposed on a policy trained with "
                    "OFFSETX_ACTION; the policy must retain control of its four x-offsets."
                )
            offset_control = "policy"
        else:
            offset = _parameter_values(
                cpg.offset_x_range,
                eval_args.offset_x_m,
                eval_args.cpg_parameter_mode,
                num_envs,
                rng,
            )
            cpg._offset_x[:] = torch.as_tensor(offset, device=env.device).unsqueeze(1).repeat(1, 4)
            offset_control = "fixed" if eval_args.cpg_parameter_mode == "nominal" else "randomized"

    return {
        "robot_height_m": height,
        "ground_clearance_m": clearance,
        "ground_penetration_m": penetration,
        "offset_x_control": offset_control,
    }


def find_spine_index(env: Any) -> int | None:
    for index, name in enumerate(env.dof_names):
        if "sp_" in name.lower() or "spine" in name.lower():
            return index
    return None


def find_leg_indices(env: Any, torch: Any, spine_index: int | None) -> Any:
    """Use the environment's canonical leg mapping when it is available."""
    mapping = getattr(env, "leg_dof_indices", None)
    if mapping:
        indices: List[int] = []
        for values in mapping.values():
            if hasattr(values, "detach"):
                values = values.detach().cpu().tolist()
            indices.extend(int(value) for value in values)
        indices = sorted(set(indices))
    else:
        indices = [index for index in range(env.num_dofs) if index != spine_index]
    return torch.tensor(indices, device=env.device, dtype=torch.long)


def control_metadata(env: Any) -> Dict[str, Any]:
    control = env.cfg.control
    return {
        "spine_control_mode": str(getattr(control, "spine_control_mode", "none")),
        "spine_phase_mode": str(getattr(control, "spine_phase_mode", "none")),
        "spine_phase_source": str(getattr(control, "spine_phase_source", "none")),
        "spine_phase_offset_rad": float(getattr(control, "spine_phase_offset", 0.0)),
        "num_actions": int(env.num_actions),
        "num_observations": int(env.num_obs),
        "num_cpgs": int(env._cpg.num_CPGs),
    }


def infer_variant_name(env: Any) -> str:
    if bool(getattr(env.cfg.asset, "spine_locked", False)):
        return "rigid"

    control = env.cfg.control
    control_mode = str(getattr(control, "spine_control_mode", "cpg")).lower()
    phase_mode = str(getattr(control, "spine_phase_mode", "uncoupled")).lower()
    source = str(getattr(control, "spine_phase_source", "rr")).lower()
    offset = float(getattr(control, "spine_phase_offset", 0.0))

    if control_mode == "direct":
        return "direct_angle"
    if phase_mode == "uncoupled":
        return "uncoupled"

    known_offsets = [
        (0.0, "0"),
        (-0.5 * math.pi, "neg_half_pi"),
        (0.5 * math.pi, "half_pi"),
        (math.pi, "pi"),
    ]
    for expected, label in known_offsets:
        if math.isclose(wrap_angle_scalar(offset - expected), 0.0, abs_tol=1.0e-6):
            return f"phase_{source}_{label}"
    return f"phase_{source}_{offset:+.4f}_rad"


def wrap_angle_scalar(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def spine_source_indices(source: str) -> List[int]:
    mapping = {
        "fl": [0],
        "fr": [1],
        "rl": [2],
        "rr": [3],
        "front_mean": [0, 1],
        "rear_mean": [2, 3],
        "left_diagonal": [0, 3],
        "right_diagonal": [1, 2],
        "all_mean": [0, 1, 2, 3],
    }
    if source not in mapping:
        raise ValueError(f"Unsupported spine phase source: {source}")
    return mapping[source]


def initialize_accumulators(torch: Any, num_envs: int, device: Any) -> Dict[str, Any]:
    zeros = lambda: torch.zeros(num_envs, dtype=torch.float, device=device)
    return {
        "count": zeros(),
        "sum_velocity": zeros(),
        "sum_abs_velocity_error": zeros(),
        "sum_sq_velocity_error": zeros(),
        "sum_phase_error": zeros(),
        "sum_roll_sq": zeros(),
        "sum_pitch_sq": zeros(),
        "energy_total": zeros(),
        "energy_leg": zeros(),
        "energy_spine": zeros(),
        "distance": zeros(),
        "leg_saturated": zeros(),
        "leg_actuator_samples": zeros(),
        "requested_leg_saturated": zeros(),
        "requested_leg_samples": zeros(),
        "peak_contact_force": zeros(),
        "sum_spine_angle_sq": zeros(),
        "spine_min": torch.full((num_envs,), float("inf"), device=device),
        "spine_max": torch.full((num_envs,), float("-inf"), device=device),
        "sum_spine_relative_sin": zeros(),
        "sum_spine_relative_cos": zeros(),
        "sum_spine_target_abs_error": zeros(),
        "spine_phase_samples": zeros(),
        "spine_target_samples": zeros(),
        "sum_offset_x_mean": zeros(),
        "offset_x_samples": zeros(),
        "fell": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }


def append_timeseries(storage: MutableMapping[str, List[np.ndarray]], **values: Any) -> None:
    for name, value in values.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        storage.setdefault(name, []).append(np.asarray(value))


def accumulate_step(
    env: Any,
    speed: float,
    valid: Any,
    accum: MutableMapping[str, Any],
    dt: float,
    leg_indices: Any,
    spine_index: int | None,
    get_euler_xyz: Any,
    contact_threshold_n: float,
    timeseries: MutableMapping[str, List[np.ndarray]] | None,
) -> None:
    import torch

    count_increment = valid.float()
    accum["count"] += count_increment

    velocity = env.base_lin_vel[:, 0]
    velocity_error = velocity - float(speed)
    accum["sum_velocity"] += torch.where(valid, velocity, 0.0)
    accum["sum_abs_velocity_error"] += torch.where(valid, torch.abs(velocity_error), 0.0)
    accum["sum_sq_velocity_error"] += torch.where(valid, velocity_error.square(), 0.0)
    accum["distance"] += torch.where(valid, torch.clamp(velocity, min=0.0) * dt, 0.0)

    roll, pitch, _ = get_euler_xyz(env.base_quat)
    roll = wrap_angle_torch(roll)
    pitch = wrap_angle_torch(pitch)
    accum["sum_roll_sq"] += torch.where(valid, roll.square(), 0.0)
    accum["sum_pitch_sq"] += torch.where(valid, pitch.square(), 0.0)

    all_phases = env._cpg.X[:, 1, :]
    phases = all_phases[:, :4]
    desired = env._cpg.PHI_batch[:, :4, :4]
    actual = phases.unsqueeze(1) - phases.unsqueeze(2)  # theta_j - theta_i
    phase_delta = wrap_angle_torch(actual - desired)
    pair_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool, device=env.device), diagonal=1)
    phase_error = torch.abs(phase_delta[:, pair_mask]).mean(dim=1)
    accum["sum_phase_error"] += torch.where(valid, phase_error, 0.0)

    control_mode = str(getattr(env.cfg.control, "spine_control_mode", "none")).lower()
    phase_mode = str(getattr(env.cfg.control, "spine_phase_mode", "none")).lower()
    spine_relative_phase = None
    if control_mode == "cpg" and all_phases.shape[1] >= 5:
        source = str(getattr(env.cfg.control, "spine_phase_source", "rr")).lower()
        source_ids = spine_source_indices(source)
        source_phases = phases[:, source_ids]
        source_phase = torch.atan2(
            torch.mean(torch.sin(source_phases), dim=1),
            torch.mean(torch.cos(source_phases), dim=1),
        )
        spine_relative_phase = wrap_angle_torch(all_phases[:, 4] - source_phase)
        accum["sum_spine_relative_sin"] += torch.where(valid, torch.sin(spine_relative_phase), 0.0)
        accum["sum_spine_relative_cos"] += torch.where(valid, torch.cos(spine_relative_phase), 0.0)
        accum["spine_phase_samples"] += valid.float()
        if phase_mode == "phase_locked":
            target_offset = float(getattr(env.cfg.control, "spine_phase_offset", 0.0))
            target_error = torch.abs(wrap_angle_torch(spine_relative_phase - target_offset))
            accum["sum_spine_target_abs_error"] += torch.where(valid, target_error, 0.0)
            accum["spine_target_samples"] += valid.float()

    applied_torque = env.torques
    joint_velocity = env.dof_vel
    joint_power = torch.abs(applied_torque * joint_velocity)
    total_power = joint_power.sum(dim=1)
    leg_power = joint_power[:, leg_indices].sum(dim=1)
    accum["energy_total"] += torch.where(valid, total_power * dt, 0.0)
    accum["energy_leg"] += torch.where(valid, leg_power * dt, 0.0)

    if spine_index is not None:
        spine_power = joint_power[:, spine_index]
        spine_angle = env.dof_pos[:, spine_index]
        accum["energy_spine"] += torch.where(valid, spine_power * dt, 0.0)
        accum["sum_spine_angle_sq"] += torch.where(valid, spine_angle.square(), 0.0)
        accum["spine_min"] = torch.where(valid, torch.minimum(accum["spine_min"], spine_angle), accum["spine_min"])
        accum["spine_max"] = torch.where(valid, torch.maximum(accum["spine_max"], spine_angle), accum["spine_max"])
    else:
        spine_angle = None

    if hasattr(env._cpg, "_offset_x"):
        offset_x = env._cpg._offset_x
        offset_x_mean = offset_x.mean(dim=1)
        accum["sum_offset_x_mean"] += torch.where(valid, offset_x_mean, 0.0)
        accum["offset_x_samples"] += valid.float()
    else:
        offset_x = None

    leg_limits = env.torque_limits[leg_indices]
    applied_saturated = torch.abs(applied_torque[:, leg_indices]) >= (leg_limits.unsqueeze(0) - 1.0e-5)
    accum["leg_saturated"] += torch.where(valid, applied_saturated.sum(dim=1).float(), 0.0)
    accum["leg_actuator_samples"] += valid.float() * int(len(leg_indices))

    requested_torque = getattr(env, "unclipped_torques", None)
    if requested_torque is None:
        requested_torque = getattr(env, "torques_unclipped", None)
    if requested_torque is None:
        requested_torque = getattr(env, "requested_torques", None)
    if requested_torque is not None:
        requested_saturated = torch.abs(requested_torque[:, leg_indices]) > leg_limits.unsqueeze(0)
        accum["requested_leg_saturated"] += torch.where(valid, requested_saturated.sum(dim=1).float(), 0.0)
        accum["requested_leg_samples"] += valid.float() * int(len(leg_indices))

    foot_forces = torch.linalg.vector_norm(env.contact_forces[:, env.feet_indices, :], dim=-1)
    peak_force = foot_forces.max(dim=1).values
    accum["peak_contact_force"] = torch.where(
        valid,
        torch.maximum(accum["peak_contact_force"], peak_force),
        accum["peak_contact_force"],
    )
    contacts = env.contact_forces[:, env.feet_indices, 2] > float(contact_threshold_n)

    if timeseries is not None:
        append_timeseries(
            timeseries,
            valid=valid,
            velocity=velocity,
            roll=roll,
            pitch=pitch,
            cpg_phase=all_phases,
            spine_relative_phase=spine_relative_phase,
            contacts=contacts,
            applied_torque=applied_torque,
            requested_torque=requested_torque,
            dof_velocity=joint_velocity,
            spine_angle=spine_angle,
            robot_height=env._cpg._robot_height,
            ground_clearance=env._cpg._ground_clearance,
            ground_penetration=env._cpg._ground_penetration,
            offset_x=offset_x,
        )


def finite_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not math.isfinite(denominator):
        return math.nan
    return numerator / denominator


def make_episode_rows(
    accum: Mapping[str, Any],
    variant: str,
    task: str,
    gait: str,
    speed: float,
    dt: float,
    robot_mass: float,
    has_spine: bool,
    control_info: Mapping[str, Any],
    cpg_parameter_mode: str,
    cpg_parameters: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    arrays = {name: value.detach().cpu().numpy() for name, value in accum.items()}
    rows: List[Dict[str, Any]] = []
    num_envs = len(arrays["count"])

    for episode in range(num_envs):
        count = float(arrays["count"][episode])
        mean_velocity = finite_divide(float(arrays["sum_velocity"][episode]), count)
        velocity_mae = finite_divide(float(arrays["sum_abs_velocity_error"][episode]), count)
        velocity_mse = finite_divide(float(arrays["sum_sq_velocity_error"][episode]), count)
        phase_error = finite_divide(float(arrays["sum_phase_error"][episode]), count)
        roll_mse = finite_divide(float(arrays["sum_roll_sq"][episode]), count)
        pitch_mse = finite_divide(float(arrays["sum_pitch_sq"][episode]), count)
        total_energy = float(arrays["energy_total"][episode])
        leg_energy = float(arrays["energy_leg"][episode])
        spine_energy = float(arrays["energy_spine"][episode]) if has_spine else math.nan
        distance = float(arrays["distance"][episode])
        measurement_time = count * dt
        mean_power = finite_divide(total_energy, measurement_time)
        cot_denominator = robot_mass * 9.81 * distance
        cot = finite_divide(total_energy, cot_denominator)
        applied_saturation = finite_divide(
            float(arrays["leg_saturated"][episode]),
            float(arrays["leg_actuator_samples"][episode]),
        )
        requested_saturation = finite_divide(
            float(arrays["requested_leg_saturated"][episode]),
            float(arrays["requested_leg_samples"][episode]),
        )

        if has_spine and count > 0:
            spine_rms = math.sqrt(max(0.0, float(arrays["sum_spine_angle_sq"][episode]) / count))
            spine_range = float(arrays["spine_max"][episode] - arrays["spine_min"][episode])
        else:
            spine_rms = math.nan
            spine_range = math.nan

        spine_phase_count = float(arrays["spine_phase_samples"][episode])
        if spine_phase_count > 0:
            mean_sin = float(arrays["sum_spine_relative_sin"][episode]) / spine_phase_count
            mean_cos = float(arrays["sum_spine_relative_cos"][episode]) / spine_phase_count
            spine_plv = math.hypot(mean_sin, mean_cos)
            spine_mean_phase = math.atan2(mean_sin, mean_cos)
        else:
            spine_plv = math.nan
            spine_mean_phase = math.nan
        spine_target_error = finite_divide(
            float(arrays["sum_spine_target_abs_error"][episode]),
            float(arrays["spine_target_samples"][episode]),
        )
        mean_offset_x = finite_divide(
            float(arrays["sum_offset_x_mean"][episode]),
            float(arrays["offset_x_samples"][episode]),
        )

        row = {
            "variant": variant,
            "task": task,
            **control_info,
            "cpg_parameter_mode": cpg_parameter_mode,
            "robot_height_m": float(cpg_parameters["robot_height_m"][episode]),
            "ground_clearance_m": float(cpg_parameters["ground_clearance_m"][episode]),
            "ground_penetration_m": float(cpg_parameters["ground_penetration_m"][episode]),
            "mean_offset_x_m": mean_offset_x,
            "offset_x_control": cpg_parameters["offset_x_control"],
            "gait": gait,
            "command_mps": speed,
            "episode": episode,
            "fell": int(bool(arrays["fell"][episode])),
            "valid_steps": int(count),
            "measurement_time_s": measurement_time,
            "mean_velocity_mps": mean_velocity,
            "velocity_bias_mps": mean_velocity - speed if math.isfinite(mean_velocity) else math.nan,
            "velocity_mae_mps": velocity_mae,
            "velocity_rmse_mps": math.sqrt(max(0.0, velocity_mse)) if math.isfinite(velocity_mse) else math.nan,
            "phase_error_rad": phase_error,
            "phase_error_deg": math.degrees(phase_error) if math.isfinite(phase_error) else math.nan,
            "roll_rms_rad": math.sqrt(max(0.0, roll_mse)) if math.isfinite(roll_mse) else math.nan,
            "pitch_rms_rad": math.sqrt(max(0.0, pitch_mse)) if math.isfinite(pitch_mse) else math.nan,
            "mean_mechanical_power_w": mean_power,
            "total_mechanical_energy_j": total_energy,
            "leg_mechanical_energy_j": leg_energy,
            "spine_mechanical_energy_j": spine_energy,
            "distance_m": distance,
            "cost_of_transport": cot,
            "leg_torque_saturation_fraction": applied_saturation,
            "requested_leg_torque_saturation_fraction": requested_saturation,
            "peak_foot_contact_force_n": float(arrays["peak_contact_force"][episode]),
            "spine_angle_rms_rad": spine_rms,
            "spine_angle_range_rad": spine_range,
            "spine_leg_phase_locking_value": spine_plv,
            "spine_leg_mean_phase_rad": spine_mean_phase,
            "spine_target_phase_error_rad": spine_target_error,
            "robot_mass_kg": robot_mass,
        }
        rows.append(row)

    return rows


def save_timeseries(path: Path, storage: Mapping[str, List[np.ndarray]], dt: float) -> None:
    if not storage:
        return
    arrays = {name: np.stack(values, axis=0) for name, values in storage.items()}
    first = next(iter(arrays.values()))
    arrays["time_s"] = np.arange(first.shape[0], dtype=np.float64) * dt
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_collection(eval_args: argparse.Namespace, legged_args: Sequence[str]) -> None:
    if eval_args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if eval_args.warmup_s < 0.0 or eval_args.measure_s <= 0.0:
        raise SystemExit("--warmup-s must be nonnegative and --measure-s must be positive")

    # Isaac Gym must be imported before torch in its supported environment.
    from isaacgym import gymapi  # noqa: F401
    from isaacgym.torch_utils import get_euler_xyz
    import torch

    import legged_gym.envs  # noqa: F401  # imports and registers tasks
    from legged_gym.utils import get_args, task_registry

    sys.argv = [sys.argv[0], *legged_args]
    args = get_args()
    args.num_envs = eval_args.episodes
    args.resume = True

    random.seed(eval_args.eval_seed)
    np.random.seed(eval_args.eval_seed)
    torch.manual_seed(eval_args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_args.eval_seed)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_evaluation_environment(env_cfg, eval_args)
    train_cfg.runner.resume = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    detected_variant = infer_variant_name(env)
    variant = eval_args.variant or detected_variant
    if eval_args.variant and safe_name(eval_args.variant) != safe_name(detected_variant):
        print(
            f"Note: --variant={eval_args.variant!r}, while the loaded control config "
            f"looks like {detected_variant!r}. Keeping the explicit label."
        )
    variant_dir = eval_args.output_dir / safe_name(variant)
    summary_path = variant_dir / "summary.csv"
    if summary_path.exists() and not eval_args.overwrite:
        raise SystemExit(f"{summary_path} already exists; pass --overwrite to replace it")
    variant_dir.mkdir(parents=True, exist_ok=True)

    robot_mass = get_robot_mass(env, eval_args.robot_mass_kg)
    dt = float(env.dt)
    warmup_steps = int(round(eval_args.warmup_s / dt))
    measure_steps = int(round(eval_args.measure_s / dt))
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    spine_index = find_spine_index(env)
    has_spine = spine_index is not None and not bool(getattr(env.cfg.asset, "spine_locked", False))
    leg_indices = find_leg_indices(env, torch, spine_index)
    control_info = control_metadata(env)
    cpg_rng = np.random.default_rng(eval_args.eval_seed)
    if (
        not hasattr(env, "unclipped_torques")
        and not hasattr(env, "torques_unclipped")
        and not hasattr(env, "requested_torques")
    ):
        print(
            "Note: the environment does not expose pre-clipping torques; "
            "requested_leg_torque_saturation_fraction will be NaN. "
            "Applied saturation is still recorded."
        )

    all_rows: List[Dict[str, Any]] = []
    for gait in [name.upper() for name in eval_args.gaits]:
        for speed in eval_args.speeds:
            env.reset_idx(all_ids)
            set_fixed_gait(env, gait)
            set_fixed_command(env, speed)
            cpg_parameters = set_cpg_episode_parameters(env, eval_args, cpg_rng, torch)
            env.compute_observations()
            obs = env.get_observations()

            accum = initialize_accumulators(torch, env.num_envs, env.device)
            active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            timeseries: Dict[str, List[np.ndarray]] | None = {} if eval_args.save_timeseries else None

            total_steps = warmup_steps + measure_steps
            for step in range(total_steps):
                set_fixed_command(env, speed)
                with torch.no_grad():
                    actions = policy(obs.detach())
                    obs, _, _, dones, _ = env.step(actions.detach())

                dones = dones.bool()
                valid = active & (~dones)
                accum["fell"] |= active & dones

                if step >= warmup_steps:
                    accumulate_step(
                        env=env,
                        speed=float(speed),
                        valid=valid,
                        accum=accum,
                        dt=dt,
                        leg_indices=leg_indices,
                        spine_index=spine_index if has_spine else None,
                        get_euler_xyz=get_euler_xyz,
                        contact_threshold_n=eval_args.contact_threshold_n,
                        timeseries=timeseries,
                    )

                active &= ~dones

            rows = make_episode_rows(
                accum=accum,
                variant=variant,
                task=args.task,
                gait=gait,
                speed=float(speed),
                dt=dt,
                robot_mass=robot_mass,
                has_spine=has_spine,
                control_info=control_info,
                cpg_parameter_mode=eval_args.cpg_parameter_mode,
                cpg_parameters=cpg_parameters,
            )
            all_rows.extend(rows)

            if timeseries is not None:
                filename = f"{safe_name(gait)}_{float(speed):.2f}mps.npz"
                save_timeseries(variant_dir / "timeseries" / filename, timeseries, dt)

            valid_rmse = [row["velocity_rmse_mps"] for row in rows if math.isfinite(row["velocity_rmse_mps"])]
            mean_rmse = statistics.fmean(valid_rmse) if valid_rmse else math.nan
            fall_rate = statistics.fmean(float(row["fell"]) for row in rows)
            print(
                f"[{variant}] {gait:20s} {float(speed):.2f} m/s "
                f"RMSE={mean_rmse:.3f} m/s, fall_rate={fall_rate:.3f}"
            )

    write_csv(summary_path, all_rows, SUMMARY_FIELDS)
    metadata = {
        "variant": variant,
        "detected_variant": detected_variant,
        "task": args.task,
        **control_info,
        "gaits": [name.upper() for name in eval_args.gaits],
        "speeds_mps": list(map(float, eval_args.speeds)),
        "episodes_per_condition": eval_args.episodes,
        "warmup_s": eval_args.warmup_s,
        "measure_s": eval_args.measure_s,
        "dt_s": dt,
        "eval_seed": eval_args.eval_seed,
        "domain_randomization_kept": eval_args.keep_domain_rand,
        "cpg_parameter_mode": eval_args.cpg_parameter_mode,
        "robot_height_range_m": list(map(float, env._cpg.robot_height_range)),
        "ground_clearance_range_m": list(map(float, env._cpg.ground_clearance_range)),
        "ground_penetration_range_m": list(map(float, env._cpg.ground_penetration_range)),
        "offset_x_range_m": list(map(float, env._cpg.offset_x_range)),
        "offset_x_from_policy_actions": bool(getattr(env._cpg, "_offset_x_from_actions", False)),
        "robot_mass_kg": robot_mass,
        "load_run": getattr(args, "load_run", None),
        "checkpoint": getattr(args, "checkpoint", None),
    }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(all_rows)} episode summaries to {summary_path}")


def read_summary_files(input_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(input_dir.glob("**/summary.csv")):
        if path.name == "combined_summary.csv":
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def aggregate_rows(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups: Dict[Tuple[str, str, float], List[Mapping[str, str]]] = {}
    for row in rows:
        key = (row["variant"], row["gait"], to_float(row["command_mps"]))
        groups.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    fields = ["variant", "gait", "command_mps", "episodes"]
    for metric in AGGREGATE_METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_ci95"])

    for (variant, gait, speed), group in sorted(groups.items()):
        result: Dict[str, Any] = {
            "variant": variant,
            "gait": gait,
            "command_mps": speed,
            "episodes": len(group),
        }
        for metric in AGGREGATE_METRICS:
            values = [to_float(row.get(metric)) for row in group]
            values = [value for value in values if math.isfinite(value)]
            if values:
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            else:
                mean = std = ci95 = math.nan
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_ci95"] = ci95
        output.append(result)

    return output, fields


def make_comparison_plots(aggregate: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    matplotlib_config = output_dir / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; aggregate CSV files were still created")
        return

    variants = sorted({str(row["variant"]) for row in aggregate})
    gaits = [gait for gait in DEFAULT_GAITS if gait in {str(row["gait"]) for row in aggregate}]
    other_gaits = sorted({str(row["gait"]) for row in aggregate} - set(gaits))
    gaits.extend(other_gaits)
    lookup = {
        (str(row["variant"]), str(row["gait"]), float(row["command_mps"])): row
        for row in aggregate
    }

    def plot_metric(metric: str, ylabel: str, filename: str, identity: bool = False) -> None:
        columns = 3
        rows_n = int(math.ceil(len(gaits) / columns))
        fig, axes = plt.subplots(rows_n, columns, figsize=(13, 3.7 * rows_n), squeeze=False)
        for axis, gait in zip(axes.flat, gaits):
            for variant in variants:
                points = [
                    row for key, row in lookup.items()
                    if key[0] == variant and key[1] == gait
                ]
                points.sort(key=lambda row: float(row["command_mps"]))
                if not points:
                    continue
                x = [float(row["command_mps"]) for row in points]
                y = [to_float(row.get(f"{metric}_mean")) for row in points]
                ci = [to_float(row.get(f"{metric}_ci95")) for row in points]
                axis.errorbar(x, y, yerr=ci, marker="o", capsize=2, label=variant)
            if identity:
                commands = [float(row["command_mps"]) for row in aggregate]
                limits = [min(commands), max(commands)]
                axis.plot(limits, limits, "k--", linewidth=1, label="ideal")
            axis.set_title(gait.replace("_", " ").title())
            axis.set_xlabel("Commanded velocity [m/s]")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
        for axis in axes.flat[len(gaits):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    plot_metric("mean_velocity_mps", "Measured velocity [m/s]", "velocity_tracking.png", identity=True)
    plot_metric("velocity_rmse_mps", "Velocity RMSE [m/s]", "velocity_rmse.png")
    plot_metric("phase_error_deg", "CPG phase error [deg]", "phase_error.png")
    plot_metric(
        "leg_torque_saturation_fraction",
        "Applied torque saturation fraction",
        "torque_saturation.png",
    )
    plot_metric(
        "requested_leg_torque_saturation_fraction",
        "Requested torque saturation fraction",
        "requested_torque_saturation.png",
    )
    plot_metric(
        "spine_leg_phase_locking_value",
        "Spine-leg phase-locking value",
        "spine_phase_locking.png",
    )
    plot_metric(
        "spine_target_phase_error_rad",
        "Spine target phase error [rad]",
        "spine_target_phase_error.png",
    )


def run_comparison(eval_args: argparse.Namespace) -> None:
    input_dir = eval_args.input_dir or eval_args.output_dir
    rows = read_summary_files(input_dir)
    if not rows:
        raise SystemExit(f"No variant summary.csv files found under {input_dir}")

    output_dir = eval_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "combined_summary.csv", rows, SUMMARY_FIELDS)
    aggregate, fields = aggregate_rows(rows)
    write_csv(output_dir / "aggregate_by_variant_gait_speed.csv", aggregate, fields)
    make_comparison_plots(aggregate, output_dir)
    print(f"Combined {len(rows)} episode rows from {len(set(row['variant'] for row in rows))} variants")
    print(f"Saved comparison results to {output_dir}")


def main() -> None:
    eval_args, remaining = parse_evaluation_args()
    if eval_args.eval_mode == "collect":
        run_collection(eval_args, remaining)
    else:
        run_comparison(eval_args)


if __name__ == "__main__":
    main()
