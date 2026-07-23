#!/usr/bin/env python3
"""Gait-transition evaluation for trained Silver Badger policies.

The evaluator is a companion to ``steady_state_evaluation.py``.  In ``collect``
mode it loads one policy through the normal legged_gym task registry, lets the
robot establish a source gait at a fixed forward-speed command, changes only
the CPG gait matrix, and measures the transient without resetting the robot or
oscillator state.  In ``compare`` mode it combines runs from several controller
variants and writes aggregate tables and figures.

Transition completion is reported separately for:

* internal CPG phase convergence;
* physical foot-touchdown pattern convergence; and
* forward-velocity recovery.

The touchdown classifier is independent of the leg CPG phase differences.  It
uses the phase of the FL oscillator only as a cycle clock and compares the
relative phases of physical touchdown events with every gait template.  This
allows the script to detect cases where the oscillator state and realized gait
do not agree.

All evaluator-specific options are parsed here.  Remaining options, including
``--task``, ``--load_run``, ``--checkpoint``, and ``--headless``, are forwarded
to ``legged_gym.utils.get_args``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from steady_state_evaluation import (
    control_metadata,
    find_leg_indices,
    find_spine_index,
    get_robot_mass,
    infer_variant_name,
    safe_name,
    set_cpg_episode_parameters,
    set_fixed_command,
    set_fixed_gait,
    set_if_present,
    wrap_angle_torch,
    write_csv,
)


LEG_NAMES = ("FL", "FR", "RL", "RR")
DEFAULT_GAITS = (
    "TROT",
    "WALK",
    "PACE",
    "BOUND",
    "PRONK",
    "CANTER",
    "TRANSVERSE_GALLOP",
    "ROTARY_GALLOP",
    "AMBLE",
)
DEFAULT_TRANSITIONS = (
    "WALK:TROT",
    "TROT:WALK",
    "TROT:BOUND",
    "BOUND:TROT",
    "TROT:ROTARY_GALLOP",
    "ROTARY_GALLOP:TROT",
)
DEFAULT_SPEEDS = tuple(round(0.2 * index, 1) for index in range(1, 11))


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
    "source_gait",
    "target_gait",
    "transition",
    "command_mps",
    "episode",
    "condition_seed",
    "cpg_parameter_mode",
    "robot_height_m",
    "ground_clearance_m",
    "ground_penetration_m",
    "offset_x_control",
    "foot_order_source",
    "fell_before_switch",
    "fell_after_switch",
    "fall_time_after_switch_s",
    "valid_post_time_s",
    "source_cpg_ready",
    "source_contact_ready",
    "source_velocity_ready",
    "source_ready",
    "source_phase_error_mean_deg",
    "source_phase_error_final_deg",
    "source_contact_realized_gait",
    "source_contact_match_fraction",
    "source_contact_score_deg",
    "source_contact_margin_deg",
    "pre_mean_velocity_mps",
    "pre_velocity_rmse_mps",
    "pre_velocity_final_within_tolerance_fraction",
    "cpg_target_error_at_switch_deg",
    "cpg_target_error_peak_deg",
    "cpg_target_error_final_deg",
    "cpg_target_error_iae_deg_s",
    "cpg_transition_completed",
    "cpg_transition_time_s",
    "contact_transition_completed",
    "contact_transition_time_s",
    "target_contact_stable",
    "target_contact_final_realized_gait",
    "target_contact_match_fraction",
    "target_contact_score_deg",
    "target_contact_margin_deg",
    "touchdowns_fl",
    "touchdowns_fr",
    "touchdowns_rl",
    "touchdowns_rr",
    "velocity_recovered",
    "velocity_recovery_time_s",
    "velocity_recovery_tolerance_mps",
    "velocity_final_stable",
    "velocity_final_within_tolerance_fraction",
    "post_mean_velocity_mps",
    "post_velocity_bias_mps",
    "post_velocity_rmse_mps",
    "peak_abs_velocity_error_mps",
    "velocity_error_iae_m",
    "post_roll_rms_rad",
    "post_pitch_rms_rad",
    "peak_abs_roll_rad",
    "peak_abs_pitch_rad",
    "peak_abs_roll_deg",
    "peak_abs_pitch_deg",
    "peak_foot_contact_force_n",
    "leg_torque_saturation_fraction",
    "requested_leg_torque_saturation_fraction",
    "requested_torque_peak_ratio",
    "requested_torque_exceedance_mean_nm",
    "longest_requested_saturation_s",
    "leg_joint_tracking_rmse_rad",
    "post_mechanical_energy_j",
    "post_leg_mechanical_energy_j",
    "post_spine_mechanical_energy_j",
    "post_distance_m",
    "post_cost_of_transport",
    "post_spine_angle_rms_rad",
    "post_spine_angle_range_rad",
    "transition_success",
    "robot_mass_kg",
]


AGGREGATE_METRICS = [
    "fell_before_switch",
    "fell_after_switch",
    "source_cpg_ready",
    "source_contact_ready",
    "source_velocity_ready",
    "source_ready",
    "source_phase_error_mean_deg",
    "source_contact_match_fraction",
    "pre_velocity_rmse_mps",
    "pre_velocity_final_within_tolerance_fraction",
    "cpg_target_error_at_switch_deg",
    "cpg_target_error_peak_deg",
    "cpg_target_error_final_deg",
    "cpg_target_error_iae_deg_s",
    "cpg_transition_completed",
    "cpg_transition_time_s",
    "contact_transition_completed",
    "contact_transition_time_s",
    "target_contact_stable",
    "target_contact_match_fraction",
    "target_contact_score_deg",
    "velocity_recovered",
    "velocity_recovery_time_s",
    "velocity_final_stable",
    "velocity_final_within_tolerance_fraction",
    "post_mean_velocity_mps",
    "post_velocity_bias_mps",
    "post_velocity_rmse_mps",
    "peak_abs_velocity_error_mps",
    "velocity_error_iae_m",
    "post_roll_rms_rad",
    "post_pitch_rms_rad",
    "peak_abs_roll_deg",
    "peak_abs_pitch_deg",
    "peak_foot_contact_force_n",
    "leg_torque_saturation_fraction",
    "requested_leg_torque_saturation_fraction",
    "requested_torque_peak_ratio",
    "requested_torque_exceedance_mean_nm",
    "longest_requested_saturation_s",
    "leg_joint_tracking_rmse_rad",
    "post_mechanical_energy_j",
    "post_leg_mechanical_energy_j",
    "post_spine_mechanical_energy_j",
    "post_cost_of_transport",
    "transition_success",
]


def parse_evaluation_args(
    argv: Sequence[str] | None = None,
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-mode", choices=("collect", "compare"), default="collect")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("gait_transition_results"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--transitions", nargs="+", default=list(DEFAULT_TRANSITIONS))
    parser.add_argument("--all-transitions", action="store_true")
    parser.add_argument("--gaits", nargs="+", default=list(DEFAULT_GAITS))
    parser.add_argument("--speeds", nargs="+", type=float, default=list(DEFAULT_SPEEDS))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--source-settle-s", type=float, default=4.0)
    parser.add_argument("--pre-window-s", type=float, default=2.0)
    parser.add_argument("--post-window-s", type=float, default=8.0)
    parser.add_argument("--final-window-s", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--contact-threshold-n", type=float, default=1.0)
    parser.add_argument("--phase-threshold-deg", type=float, default=20.0)
    parser.add_argument("--phase-hold-s", type=float, default=0.4)
    parser.add_argument("--contact-score-threshold-deg", type=float, default=30.0)
    parser.add_argument("--contact-min-margin-deg", type=float, default=3.0)
    parser.add_argument("--contact-match-fraction", type=float, default=0.8)
    parser.add_argument("--contact-hold-s", type=float, default=0.3)
    parser.add_argument("--contact-lookback-s", type=float, default=2.5)
    parser.add_argument("--contact-strikes-per-foot", type=int, default=2)
    parser.add_argument("--contact-min-strike-separation-s", type=float, default=0.08)
    parser.add_argument("--velocity-abs-tolerance-mps", type=float, default=0.10)
    parser.add_argument("--velocity-rel-tolerance", type=float, default=0.10)
    parser.add_argument("--velocity-hold-s", type=float, default=0.5)
    parser.add_argument("--velocity-final-fraction", type=float, default=0.8)
    parser.add_argument("--robot-mass-kg", type=float, default=None)
    parser.add_argument(
        "--cpg-parameter-mode",
        choices=("nominal", "randomized"),
        default="nominal",
        help="Use training-range midpoints or paired deterministic samples.",
    )
    parser.add_argument("--robot-height-m", type=float, default=None)
    parser.add_argument("--ground-clearance-m", type=float, default=None)
    parser.add_argument("--ground-penetration-m", type=float, default=None)
    parser.add_argument("--offset-x-m", type=float, default=None)
    parser.add_argument("--save-timeseries", action="store_true")
    parser.add_argument("--keep-domain-rand", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--transition-help", action="store_true")
    args, remaining = parser.parse_known_args(argv)

    if args.transition_help:
        print(__doc__)
        print("Evaluator options:\n")
        parser.print_help()
        print("\nAll remaining options are passed to legged_gym.utils.get_args().")
        raise SystemExit(0)
    return args, remaining


def normalize_gait(value: str) -> str:
    return value.strip().upper().replace("-", "_").replace(" ", "_")


def parse_transition(value: str) -> Tuple[str, str]:
    normalized = value.strip().replace("->", ":").replace(",", ":")
    parts = [normalize_gait(part) for part in normalized.split(":") if part.strip()]
    if len(parts) != 2 or parts[0] == parts[1]:
        raise ValueError(
            f"Invalid transition {value!r}; use SOURCE:TARGET with different gaits."
        )
    return parts[0], parts[1]


def requested_transitions(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.all_transitions:
        gaits = [normalize_gait(gait) for gait in args.gaits]
        transitions = [(source, target) for source in gaits for target in gaits if source != target]
    else:
        transitions = [parse_transition(value) for value in args.transitions]
    return list(dict.fromkeys(transitions))


def configure_evaluation_environment(env_cfg: Any, args: argparse.Namespace) -> None:
    total_s = args.source_settle_s + args.pre_window_s + args.post_window_s
    env_cfg.env.num_envs = args.episodes
    set_if_present(env_cfg.env, "play", True)
    env_cfg.env.episode_length_s = max(float(env_cfg.env.episode_length_s), total_s + 2.0)

    set_if_present(env_cfg.commands, "curriculum", False)
    set_if_present(env_cfg.commands, "resample_gait_style", False)
    set_if_present(env_cfg.commands, "resampling_time", 1.0e9)
    set_if_present(env_cfg.commands, "gait_resampling_time", 1.0e9)
    if hasattr(env_cfg.commands, "ranges"):
        speed = float(args.speeds[0])
        set_if_present(env_cfg.commands.ranges, "lin_vel_x", [speed, speed])
        set_if_present(env_cfg.commands.ranges, "lin_vel_y", [0.0, 0.0])
        set_if_present(env_cfg.commands.ranges, "ang_vel_yaw", [0.0, 0.0])

    set_if_present(env_cfg.noise, "add_noise", False)
    if not args.keep_domain_rand:
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


def stable_condition_seed(base_seed: int, source: str, target: str, speed: float) -> int:
    payload = f"{base_seed}|{source}|{target}|{speed:.8f}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**31 - 1))


def available_gaits(env: Any) -> Dict[str, Tuple[int, Any]]:
    names = [normalize_gait(name) for name in env._cpg.gait_names]
    return {
        name: (index, env._cpg.available_gaits[index])
        for index, name in enumerate(names)
    }


def switch_gait_without_reset(env: Any, target_gait: str) -> int:
    lookup = available_gaits(env)
    target_gait = normalize_gait(target_gait)
    if target_gait not in lookup:
        raise ValueError(
            f"Gait {target_gait!r} is unavailable. Available gaits: {sorted(lookup)}"
        )
    target_index, target_matrix = lookup[target_gait]
    if hasattr(env._cpg, "previous_gait_indices"):
        env._cpg.previous_gait_indices[:] = env._cpg.current_gait_indices.clone()
    env._cpg.PHI_batch[:] = target_matrix
    env._cpg.current_gait_indices[:] = target_index
    return target_index


def phase_error(env: Any, desired_matrix: Any, torch: Any) -> Any:
    phases = env._cpg.X[:, 1, :4]
    actual = phases.unsqueeze(1) - phases.unsqueeze(2)
    desired = desired_matrix[:4, :4]
    delta = wrap_angle_torch(actual - desired.unsqueeze(0))
    pair_mask = torch.triu(
        torch.ones(4, 4, dtype=torch.bool, device=env.device), diagonal=1
    )
    return torch.abs(delta[:, pair_mask]).mean(dim=1)


def _leg_token(name: str) -> str | None:
    value = name.lower().replace("-", "_")
    padded = f"_{value}_"
    aliases = {
        "FL": ("_fl_", "front_left", "left_front", "_lf_"),
        "FR": ("_fr_", "front_right", "right_front", "_rf_"),
        "RL": ("_rl_", "rear_left", "left_rear", "hind_left", "left_hind", "_lh_"),
        "RR": ("_rr_", "rear_right", "right_rear", "hind_right", "right_hind", "_rh_"),
    }
    for leg, patterns in aliases.items():
        if any(pattern in padded for pattern in patterns):
            return leg
    return None


def canonical_foot_order(env: Any) -> Tuple[List[int], str]:
    names = getattr(env, "feet_names", None)
    if names is None or len(names) != 4:
        return [0, 1, 2, 3], "assumed_FL_FR_RL_RR"
    mapping: Dict[str, int] = {}
    for index, name in enumerate(names):
        token = _leg_token(str(name))
        if token is not None and token not in mapping:
            mapping[token] = index
    if set(mapping) == set(LEG_NAMES):
        return [mapping[name] for name in LEG_NAMES], "inferred_from_feet_names"
    return [0, 1, 2, 3], "assumed_FL_FR_RL_RR"


def gait_template_offsets(env: Any) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    offsets: List[np.ndarray] = []
    for name, matrix in zip(env._cpg.gait_names, env._cpg.available_gaits):
        names.append(normalize_gait(name))
        values = matrix[0, :4].detach().cpu().numpy().astype(np.float64)
        offsets.append(np.arctan2(np.sin(values), np.cos(values)))
    return names, np.stack(offsets, axis=0)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def append_trace(storage: MutableMapping[str, List[np.ndarray]], **values: Any) -> None:
    for name, value in values.items():
        if value is not None:
            storage.setdefault(name, []).append(_to_numpy(value).copy())


def stack_trace(storage: Mapping[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {name: np.stack(values, axis=0) for name, values in storage.items()}


def get_requested_torques(env: Any) -> Any | None:
    for name in ("unclipped_torques", "torques_unclipped", "requested_torques"):
        value = getattr(env, name, None)
        if value is not None:
            return value
    return None


def record_state(
    env: Any,
    actions: Any,
    valid: Any,
    source_matrix: Any,
    target_matrix: Any,
    foot_order: Sequence[int],
    get_euler_xyz: Any,
    contact_threshold_n: float,
    torch: Any,
) -> Dict[str, Any]:
    roll, pitch, _ = get_euler_xyz(env.base_quat)
    roll = wrap_angle_torch(roll)
    pitch = wrap_angle_torch(pitch)
    ordered_feet = env.feet_indices[list(foot_order)]
    foot_force_vectors = env.contact_forces[:, ordered_feet, :]
    foot_force_norms = torch.linalg.vector_norm(foot_force_vectors, dim=-1)
    contacts = foot_force_vectors[:, :, 2] > float(contact_threshold_n)
    spine_index = find_spine_index(env)
    spine_angle = env.dof_pos[:, spine_index] if spine_index is not None else None
    return {
        "valid": valid,
        "velocity_mps": env.base_lin_vel[:, 0],
        "roll_rad": roll,
        "pitch_rad": pitch,
        "cpg_phase_rad": env._cpg.X[:, 1, :],
        "cpg_amplitude": env._cpg.X[:, 0, :],
        "cpg_phase_velocity_rads": env._cpg.X_dot[:, 1, :],
        "source_phase_error_rad": phase_error(env, source_matrix, torch),
        "target_phase_error_rad": phase_error(env, target_matrix, torch),
        "contacts": contacts,
        "foot_contact_force_n": foot_force_norms,
        "actions": actions,
        "applied_torque_nm": env.torques,
        "requested_torque_nm": get_requested_torques(env),
        "dof_velocity_rads": env.dof_vel,
        "dof_position_rad": env.dof_pos,
        "dof_desired_position_rad": getattr(env, "dof_des_pos", None),
        "spine_angle_rad": spine_angle,
        "offset_x_m": getattr(env._cpg, "_offset_x", None),
    }


def finite_values(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(array)
    if valid is not None:
        valid_array = np.asarray(valid, dtype=bool)
        while valid_array.ndim < array.ndim:
            valid_array = valid_array[..., None]
        mask &= np.broadcast_to(valid_array, array.shape)
    return array[mask]


def safe_mean(values: np.ndarray, valid: np.ndarray | None = None) -> float:
    selected = finite_values(values, valid)
    return float(np.mean(selected)) if selected.size else math.nan


def safe_rms(values: np.ndarray, valid: np.ndarray | None = None) -> float:
    selected = finite_values(values, valid)
    return float(np.sqrt(np.mean(np.square(selected)))) if selected.size else math.nan


def safe_peak_abs(values: np.ndarray, valid: np.ndarray | None = None) -> float:
    selected = finite_values(values, valid)
    return float(np.max(np.abs(selected))) if selected.size else math.nan


def safe_final_mean(
    values: np.ndarray,
    valid: np.ndarray,
    window_steps: int,
) -> float:
    valid_indices = np.flatnonzero(np.asarray(valid, dtype=bool))
    if valid_indices.size == 0:
        return math.nan
    end = int(valid_indices[-1]) + 1
    start = max(0, end - max(1, window_steps))
    return safe_mean(np.asarray(values)[start:end], np.asarray(valid)[start:end])


def first_sustained_true(mask: np.ndarray, hold_steps: int) -> int | None:
    run = 0
    required = max(1, int(hold_steps))
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        run = run + 1 if value else 0
        if run >= required:
            return index - required + 1
    return None


def longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def circular_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.arctan2(np.mean(np.sin(array)), np.mean(np.cos(array))))


def circular_residual(values: np.ndarray) -> Tuple[np.ndarray, float]:
    center = circular_mean(values)
    residual = np.arctan2(np.sin(values - center), np.cos(values - center))
    return residual, center


def touchdown_classification(
    reference_phase: np.ndarray,
    contacts: np.ndarray,
    valid: np.ndarray,
    gait_offsets: np.ndarray,
    lookback_steps: int,
    strikes_per_foot: int,
    min_separation_steps: int,
    initial_contacts: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Classify realized gait from recent physical touchdown phases.

    If leg ``j`` touches down at a common leg-oscillator phase, the FL clock
    phase at that event is ``common_phase - gait_offset[j]``.  Adding a
    candidate offset to each observed touchdown phase should therefore align
    all four values.  The mean absolute circular residual is the candidate's
    score; lower is better.
    """
    reference_phase = np.asarray(reference_phase, dtype=np.float64)
    contacts = np.asarray(contacts, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    gait_offsets = np.asarray(gait_offsets, dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[1] != 4:
        raise ValueError(f"Expected contacts with shape [time, 4], got {contacts.shape}")

    steps = contacts.shape[0]
    gait_count = gait_offsets.shape[0]
    scores = np.full((steps, gait_count), np.nan, dtype=np.float64)
    best_index = np.full(steps, -1, dtype=np.int64)
    margin = np.full(steps, np.nan, dtype=np.float64)
    touchdown = np.zeros((steps, 4), dtype=bool)
    event_lists: List[List[Tuple[int, float]]] = [[] for _ in range(4)]
    last_event = np.full(4, -10**9, dtype=np.int64)

    if initial_contacts is None:
        previous = contacts[0].copy() if steps else np.zeros(4, dtype=bool)
        start = 1
    else:
        previous = np.asarray(initial_contacts, dtype=bool).copy()
        start = 0

    for step in range(start, steps):
        if not valid[step]:
            previous = contacts[step].copy()
            continue

        rising = contacts[step] & ~previous
        for foot in range(4):
            if rising[foot] and step - int(last_event[foot]) >= max(1, min_separation_steps):
                event_lists[foot].append((step, float(reference_phase[step])))
                last_event[foot] = step
                touchdown[step, foot] = True
            cutoff = step - max(1, lookback_steps)
            event_lists[foot] = [event for event in event_lists[foot] if event[0] >= cutoff]
        previous = contacts[step].copy()

        if not all(len(events) >= strikes_per_foot for events in event_lists):
            continue

        observed = np.array(
            [
                circular_mean([phase for _, phase in events[-strikes_per_foot:]])
                for events in event_lists
            ],
            dtype=np.float64,
        )
        for gait_index in range(gait_count):
            aligned = observed + gait_offsets[gait_index]
            residual, _ = circular_residual(aligned)
            scores[step, gait_index] = float(np.mean(np.abs(residual)))
        order = np.argsort(scores[step])
        best_index[step] = int(order[0])
        if gait_count > 1:
            margin[step] = float(scores[step, order[1]] - scores[step, order[0]])

    return {
        "scores_rad": scores,
        "best_index": best_index,
        "margin_rad": margin,
        "touchdown": touchdown,
    }


def dominant_classification(
    classification: Mapping[str, np.ndarray],
    desired_index: int,
    valid: np.ndarray,
    window_steps: int,
) -> Tuple[int, float, float, float]:
    best = np.asarray(classification["best_index"], dtype=np.int64)
    scores = np.asarray(classification["scores_rad"], dtype=np.float64)
    margins = np.asarray(classification["margin_rad"], dtype=np.float64)
    valid_indices = np.flatnonzero(np.asarray(valid, dtype=bool))
    if valid_indices.size == 0:
        return -1, math.nan, math.nan, math.nan
    end = int(valid_indices[-1]) + 1
    start = max(0, end - max(1, window_steps))
    available = (best[start:end] >= 0) & np.asarray(valid[start:end], dtype=bool)
    if not np.any(available):
        return -1, math.nan, math.nan, math.nan
    labels = best[start:end][available]
    counts = np.bincount(labels)
    dominant = int(np.argmax(counts))
    desired_fraction = float(np.mean(labels == desired_index))
    desired_scores = scores[start:end, desired_index][available]
    score = float(np.nanmedian(desired_scores)) if np.any(np.isfinite(desired_scores)) else math.nan
    selected_margins = margins[start:end][available]
    margin = float(np.nanmedian(selected_margins)) if np.any(np.isfinite(selected_margins)) else math.nan
    return dominant, desired_fraction, score, margin


def _episode_trace(trace: Mapping[str, np.ndarray], episode: int) -> Dict[str, np.ndarray]:
    return {name: np.asarray(value)[:, episode] for name, value in trace.items()}


def _requested_saturation_metrics(
    requested: np.ndarray | None,
    limits: np.ndarray,
    valid: np.ndarray,
    dt: float,
) -> Tuple[float, float, float, float]:
    if requested is None:
        return math.nan, math.nan, math.nan, math.nan
    requested = np.asarray(requested, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return math.nan, math.nan, math.nan, math.nan
    saturation = np.abs(requested) > limits[None, :]
    samples = saturation[valid]
    fraction = float(np.mean(samples)) if samples.size else math.nan
    ratios = np.abs(requested[valid]) / np.maximum(limits[None, :], 1.0e-9)
    peak_ratio = float(np.max(ratios)) if ratios.size else math.nan
    exceedance = np.maximum(np.abs(requested[valid]) - limits[None, :], 0.0)
    positive = exceedance[exceedance > 0.0]
    mean_exceedance = float(np.mean(positive)) if positive.size else 0.0
    any_saturated = saturation.any(axis=1) & valid
    longest = float(longest_true_run(any_saturated) * dt)
    return fraction, peak_ratio, mean_exceedance, longest


def analyze_episode(
    pre_trace: Mapping[str, np.ndarray],
    post_trace: Mapping[str, np.ndarray],
    episode: int,
    source_gait: str,
    target_gait: str,
    speed: float,
    gait_names: Sequence[str],
    gait_offsets: np.ndarray,
    args: argparse.Namespace,
    dt: float,
    leg_indices: np.ndarray,
    torque_limits: np.ndarray,
    spine_index: int | None,
    robot_mass: float,
    target_error_at_switch: float,
    fell_before_switch: bool,
    fell_after_switch: bool,
    fall_step_after_switch: int,
) -> Dict[str, Any]:
    pre = _episode_trace(pre_trace, episode)
    post = _episode_trace(post_trace, episode)
    pre_valid = np.asarray(pre["valid"], dtype=bool)
    post_valid = np.asarray(post["valid"], dtype=bool)
    source_index = list(gait_names).index(source_gait)
    target_index = list(gait_names).index(target_gait)
    final_steps = max(1, int(round(args.final_window_s / dt)))
    phase_hold_steps = max(1, int(round(args.phase_hold_s / dt)))
    contact_hold_steps = max(1, int(round(args.contact_hold_s / dt)))
    velocity_hold_steps = max(1, int(round(args.velocity_hold_s / dt)))
    lookback_steps = max(1, int(round(args.contact_lookback_s / dt)))
    separation_steps = max(1, int(round(args.contact_min_strike_separation_s / dt)))

    pre_classification = touchdown_classification(
        reference_phase=pre["cpg_phase_rad"][:, 0],
        contacts=pre["contacts"],
        valid=pre_valid,
        gait_offsets=gait_offsets,
        lookback_steps=lookback_steps,
        strikes_per_foot=args.contact_strikes_per_foot,
        min_separation_steps=separation_steps,
    )
    initial_contacts = pre["contacts"][-1] if pre["contacts"].shape[0] else None
    post_classification = touchdown_classification(
        reference_phase=post["cpg_phase_rad"][:, 0],
        contacts=post["contacts"],
        valid=post_valid,
        gait_offsets=gait_offsets,
        lookback_steps=lookback_steps,
        strikes_per_foot=args.contact_strikes_per_foot,
        min_separation_steps=separation_steps,
        initial_contacts=initial_contacts,
    )

    source_label, source_fraction, source_score, source_margin = dominant_classification(
        pre_classification, source_index, pre_valid, final_steps
    )
    target_label, target_fraction, target_score, target_margin = dominant_classification(
        post_classification, target_index, post_valid, final_steps
    )

    phase_threshold = math.radians(args.phase_threshold_deg)
    contact_threshold = math.radians(args.contact_score_threshold_deg)
    contact_margin_threshold = math.radians(args.contact_min_margin_deg)
    velocity_tolerance = max(
        float(args.velocity_abs_tolerance_mps),
        abs(float(speed)) * float(args.velocity_rel_tolerance),
    )
    source_phase_final = safe_final_mean(
        pre["source_phase_error_rad"], pre_valid, phase_hold_steps
    )
    source_cpg_ready = bool(
        not fell_before_switch
        and math.isfinite(source_phase_final)
        and source_phase_final <= phase_threshold
    )
    source_contact_ready = bool(
        not fell_before_switch
        and source_label == source_index
        and math.isfinite(source_fraction)
        and source_fraction >= args.contact_match_fraction
        and math.isfinite(source_score)
        and source_score <= contact_threshold
        and math.isfinite(source_margin)
        and source_margin >= contact_margin_threshold
    )
    pre_velocity_error = np.asarray(pre["velocity_mps"], dtype=np.float64) - float(speed)
    pre_velocity_ok = (
        pre_valid
        & np.isfinite(pre_velocity_error)
        & (np.abs(pre_velocity_error) <= velocity_tolerance)
    )
    pre_final_valid = pre_valid[-final_steps:]
    pre_velocity_final_fraction = (
        float(np.mean(pre_velocity_ok[-final_steps:][pre_final_valid]))
        if np.any(pre_final_valid)
        else math.nan
    )
    source_velocity_ready = bool(
        not fell_before_switch
        and math.isfinite(pre_velocity_final_fraction)
        and pre_velocity_final_fraction >= args.velocity_final_fraction
    )
    # Eligibility is based on the realized physical gait and velocity.  CPG
    # convergence remains an independent diagnostic because the two can differ.
    source_ready = source_contact_ready and source_velocity_ready

    target_phase_error = np.asarray(post["target_phase_error_rad"], dtype=np.float64)
    phase_ok = post_valid & np.isfinite(target_phase_error) & (target_phase_error <= phase_threshold)
    phase_index = first_sustained_true(phase_ok, phase_hold_steps)
    cpg_completed = phase_index is not None
    cpg_time = (phase_index + 1) * dt if phase_index is not None else math.nan

    best = post_classification["best_index"]
    scores = post_classification["scores_rad"][:, target_index]
    margins = post_classification["margin_rad"]
    contact_ok = (
        post_valid
        & (best == target_index)
        & np.isfinite(scores)
        & (scores <= contact_threshold)
        & np.isfinite(margins)
        & (margins >= contact_margin_threshold)
    )
    contact_index = first_sustained_true(contact_ok, contact_hold_steps)
    contact_completed = contact_index is not None
    contact_time = (contact_index + 1) * dt if contact_index is not None else math.nan
    target_contact_stable = bool(
        target_label == target_index
        and math.isfinite(target_fraction)
        and target_fraction >= args.contact_match_fraction
        and math.isfinite(target_score)
        and target_score <= contact_threshold
        and math.isfinite(target_margin)
        and target_margin >= contact_margin_threshold
    )

    velocity_error = np.asarray(post["velocity_mps"], dtype=np.float64) - float(speed)
    velocity_ok = post_valid & np.isfinite(velocity_error) & (np.abs(velocity_error) <= velocity_tolerance)
    velocity_index = first_sustained_true(velocity_ok, velocity_hold_steps)
    velocity_recovered = velocity_index is not None
    velocity_time = (velocity_index + 1) * dt if velocity_index is not None else math.nan
    final_valid = post_valid[-final_steps:]
    final_velocity_ok = velocity_ok[-final_steps:]
    velocity_final_fraction = (
        float(np.mean(final_velocity_ok[final_valid])) if np.any(final_valid) else math.nan
    )
    velocity_final_stable = bool(
        math.isfinite(velocity_final_fraction)
        and velocity_final_fraction >= args.velocity_final_fraction
    )

    applied_leg = np.asarray(post["applied_torque_nm"], dtype=np.float64)[:, leg_indices]
    requested_leg = None
    if "requested_torque_nm" in post:
        requested_leg = np.asarray(post["requested_torque_nm"], dtype=np.float64)[:, leg_indices]
    leg_limits = np.asarray(torque_limits, dtype=np.float64)[leg_indices]
    applied_saturation = np.abs(applied_leg) >= (leg_limits[None, :] - 1.0e-5)
    applied_fraction = float(np.mean(applied_saturation[post_valid])) if np.any(post_valid) else math.nan
    requested_fraction, peak_ratio, mean_exceedance, longest_saturation = (
        _requested_saturation_metrics(requested_leg, leg_limits, post_valid, dt)
    )

    dof_velocity = np.asarray(post["dof_velocity_rads"], dtype=np.float64)
    applied_torque = np.asarray(post["applied_torque_nm"], dtype=np.float64)
    joint_power = np.abs(applied_torque * dof_velocity)
    total_energy = float(np.sum(joint_power[post_valid]) * dt) if np.any(post_valid) else math.nan
    leg_energy = (
        float(np.sum(joint_power[post_valid][:, leg_indices]) * dt)
        if np.any(post_valid)
        else math.nan
    )
    if spine_index is not None and np.any(post_valid):
        spine_energy = float(np.sum(joint_power[post_valid][:, spine_index]) * dt)
    else:
        spine_energy = math.nan
    distance = (
        float(np.sum(np.maximum(np.asarray(post["velocity_mps"])[post_valid], 0.0)) * dt)
        if np.any(post_valid)
        else math.nan
    )
    cot = (
        total_energy / (robot_mass * 9.81 * distance)
        if math.isfinite(total_energy)
        and math.isfinite(robot_mass)
        and math.isfinite(distance)
        and robot_mass > 0.0
        and distance > 0.0
        else math.nan
    )

    joint_tracking_rmse = math.nan
    if "dof_desired_position_rad" in post:
        tracking_error = (
            np.asarray(post["dof_desired_position_rad"], dtype=np.float64)[:, leg_indices]
            - np.asarray(post["dof_position_rad"], dtype=np.float64)[:, leg_indices]
        )
        joint_tracking_rmse = safe_rms(tracking_error, post_valid)

    spine_rms = math.nan
    spine_range = math.nan
    if spine_index is not None and "spine_angle_rad" in post:
        spine_values = finite_values(post["spine_angle_rad"], post_valid)
        if spine_values.size:
            spine_rms = float(np.sqrt(np.mean(np.square(spine_values))))
            spine_range = float(np.max(spine_values) - np.min(spine_values))

    phase_values = finite_values(target_phase_error, post_valid)
    phase_peak = float(np.max(phase_values)) if phase_values.size else math.nan
    if math.isfinite(target_error_at_switch):
        phase_peak = (
            max(phase_peak, target_error_at_switch)
            if math.isfinite(phase_peak)
            else target_error_at_switch
        )
    phase_iae = float(np.sum(phase_values) * dt) if phase_values.size else math.nan
    target_final_phase = safe_final_mean(target_phase_error, post_valid, final_steps)
    foot_forces = np.asarray(post["foot_contact_force_n"], dtype=np.float64)
    force_values = finite_values(foot_forces, post_valid)
    peak_force = float(np.max(force_values)) if force_values.size else math.nan
    touchdowns = np.sum(post_classification["touchdown"] & post_valid[:, None], axis=0)
    post_time = float(np.sum(post_valid) * dt)
    fall_time = (
        float((fall_step_after_switch + 1) * dt)
        if fell_after_switch and fall_step_after_switch >= 0
        else math.nan
    )

    source_name = gait_names[source_label] if source_label >= 0 else "UNAVAILABLE"
    target_name = gait_names[target_label] if target_label >= 0 else "UNAVAILABLE"
    transition_success: float
    if not source_ready:
        transition_success = math.nan
    else:
        transition_success = float(
            (not fell_after_switch)
            and contact_completed
            and target_contact_stable
            and velocity_recovered
            and velocity_final_stable
        )

    return {
        "fell_before_switch": int(fell_before_switch),
        "fell_after_switch": int(fell_after_switch),
        "fall_time_after_switch_s": fall_time,
        "valid_post_time_s": post_time,
        "source_cpg_ready": int(source_cpg_ready),
        "source_contact_ready": int(source_contact_ready),
        "source_velocity_ready": int(source_velocity_ready),
        "source_ready": int(source_ready),
        "source_phase_error_mean_deg": math.degrees(safe_mean(pre["source_phase_error_rad"], pre_valid)),
        "source_phase_error_final_deg": math.degrees(source_phase_final),
        "source_contact_realized_gait": source_name,
        "source_contact_match_fraction": source_fraction,
        "source_contact_score_deg": math.degrees(source_score),
        "source_contact_margin_deg": math.degrees(source_margin),
        "pre_mean_velocity_mps": safe_mean(pre["velocity_mps"], pre_valid),
        "pre_velocity_rmse_mps": safe_rms(pre_velocity_error, pre_valid),
        "pre_velocity_final_within_tolerance_fraction": pre_velocity_final_fraction,
        "cpg_target_error_at_switch_deg": math.degrees(target_error_at_switch),
        "cpg_target_error_peak_deg": math.degrees(phase_peak),
        "cpg_target_error_final_deg": math.degrees(target_final_phase),
        "cpg_target_error_iae_deg_s": math.degrees(phase_iae),
        "cpg_transition_completed": int(cpg_completed),
        "cpg_transition_time_s": cpg_time,
        "contact_transition_completed": int(contact_completed),
        "contact_transition_time_s": contact_time,
        "target_contact_stable": int(target_contact_stable),
        "target_contact_final_realized_gait": target_name,
        "target_contact_match_fraction": target_fraction,
        "target_contact_score_deg": math.degrees(target_score),
        "target_contact_margin_deg": math.degrees(target_margin),
        "touchdowns_fl": int(touchdowns[0]),
        "touchdowns_fr": int(touchdowns[1]),
        "touchdowns_rl": int(touchdowns[2]),
        "touchdowns_rr": int(touchdowns[3]),
        "velocity_recovered": int(velocity_recovered),
        "velocity_recovery_time_s": velocity_time,
        "velocity_recovery_tolerance_mps": velocity_tolerance,
        "velocity_final_stable": int(velocity_final_stable),
        "velocity_final_within_tolerance_fraction": velocity_final_fraction,
        "post_mean_velocity_mps": safe_mean(post["velocity_mps"], post_valid),
        "post_velocity_bias_mps": safe_mean(post["velocity_mps"], post_valid) - speed,
        "post_velocity_rmse_mps": safe_rms(velocity_error, post_valid),
        "peak_abs_velocity_error_mps": safe_peak_abs(velocity_error, post_valid),
        "velocity_error_iae_m": (
            float(np.sum(np.abs(velocity_error[post_valid])) * dt)
            if np.any(post_valid)
            else math.nan
        ),
        "post_roll_rms_rad": safe_rms(post["roll_rad"], post_valid),
        "post_pitch_rms_rad": safe_rms(post["pitch_rad"], post_valid),
        "peak_abs_roll_rad": safe_peak_abs(post["roll_rad"], post_valid),
        "peak_abs_pitch_rad": safe_peak_abs(post["pitch_rad"], post_valid),
        "peak_abs_roll_deg": math.degrees(safe_peak_abs(post["roll_rad"], post_valid)),
        "peak_abs_pitch_deg": math.degrees(safe_peak_abs(post["pitch_rad"], post_valid)),
        "peak_foot_contact_force_n": peak_force,
        "leg_torque_saturation_fraction": applied_fraction,
        "requested_leg_torque_saturation_fraction": requested_fraction,
        "requested_torque_peak_ratio": peak_ratio,
        "requested_torque_exceedance_mean_nm": mean_exceedance,
        "longest_requested_saturation_s": longest_saturation,
        "leg_joint_tracking_rmse_rad": joint_tracking_rmse,
        "post_mechanical_energy_j": total_energy,
        "post_leg_mechanical_energy_j": leg_energy,
        "post_spine_mechanical_energy_j": spine_energy,
        "post_distance_m": distance,
        "post_cost_of_transport": cot,
        "post_spine_angle_rms_rad": spine_rms,
        "post_spine_angle_range_rad": spine_range,
        "transition_success": transition_success,
    }


def save_transition_timeseries(
    path: Path,
    pre_trace: Mapping[str, np.ndarray],
    post_trace: Mapping[str, np.ndarray],
    dt: float,
    source_gait: str,
    target_gait: str,
    speed: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {}
    common = sorted(set(pre_trace) & set(post_trace))
    for name in common:
        arrays[name] = np.concatenate((pre_trace[name], post_trace[name]), axis=0)
    pre_steps = next(iter(pre_trace.values())).shape[0]
    post_steps = next(iter(post_trace.values())).shape[0]
    arrays["time_s"] = np.concatenate(
        (
            np.arange(-pre_steps, 0, dtype=np.float64) * dt,
            np.arange(1, post_steps + 1, dtype=np.float64) * dt,
        )
    )
    arrays["transition_index"] = np.asarray(pre_steps, dtype=np.int64)
    arrays["source_gait"] = np.asarray(source_gait)
    arrays["target_gait"] = np.asarray(target_gait)
    arrays["command_mps"] = np.asarray(speed, dtype=np.float64)
    np.savez_compressed(path, **arrays)


def run_collection(args: argparse.Namespace, legged_args: Sequence[str]) -> None:
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    for name in ("source_settle_s", "pre_window_s", "post_window_s"):
        if float(getattr(args, name)) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.contact_strikes_per_foot < 1:
        raise SystemExit("--contact-strikes-per-foot must be at least 1")
    for name in ("contact_match_fraction", "velocity_final_fraction"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if any(speed <= 0.0 for speed in args.speeds):
        raise SystemExit("All --speeds must be positive")

    # Isaac Gym must be imported before torch in supported installations.
    from isaacgym import gymapi  # noqa: F401
    from isaacgym.torch_utils import get_euler_xyz
    import torch

    import legged_gym.envs  # noqa: F401  # task registration side effect
    from legged_gym.utils import get_args, task_registry

    sys.argv = [sys.argv[0], *legged_args]
    legged = get_args()
    legged.num_envs = args.episodes
    legged.resume = True

    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)

    env_cfg, train_cfg = task_registry.get_cfgs(name=legged.task)
    configure_evaluation_environment(env_cfg, args)
    train_cfg.runner.resume = True
    env, _ = task_registry.make_env(name=legged.task, args=legged, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=legged.task, args=legged, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    detected_variant = infer_variant_name(env)
    variant = args.variant or detected_variant
    if args.variant and safe_name(args.variant) != safe_name(detected_variant):
        print(
            f"Note: --variant={args.variant!r}, while the loaded control config "
            f"looks like {detected_variant!r}. Keeping the explicit label."
        )
    variant_dir = args.output_dir / safe_name(variant)
    summary_path = variant_dir / "summary.csv"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"{summary_path} already exists; pass --overwrite to replace it")
    variant_dir.mkdir(parents=True, exist_ok=True)

    transitions = requested_transitions(args)
    gait_lookup = available_gaits(env)
    missing = sorted(
        {gait for pair in transitions for gait in pair if gait not in gait_lookup}
    )
    if missing:
        raise SystemExit(
            f"Unavailable gaits requested: {missing}. Available: {sorted(gait_lookup)}"
        )

    dt = float(env.dt)
    settle_steps = int(round(args.source_settle_s / dt))
    pre_steps = int(round(args.pre_window_s / dt))
    post_steps = int(round(args.post_window_s / dt))
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    spine_index = find_spine_index(env)
    has_active_spine = (
        spine_index is not None and not bool(getattr(env.cfg.asset, "spine_locked", False))
    )
    leg_indices_tensor = find_leg_indices(env, torch, spine_index)
    leg_indices = leg_indices_tensor.detach().cpu().numpy().astype(np.int64)
    torque_limits = env.torque_limits.detach().cpu().numpy().astype(np.float64)
    robot_mass = get_robot_mass(env, args.robot_mass_kg)
    foot_order, foot_order_source = canonical_foot_order(env)
    gait_names, gait_offsets = gait_template_offsets(env)
    control_info = control_metadata(env)

    if get_requested_torques(env) is None:
        print(
            "Note: pre-clipping torque is not exposed by this environment. "
            "Requested-torque fields will be NaN; applied saturation is still recorded."
        )
    if foot_order_source.startswith("assumed"):
        print(
            "Note: feet_names was unavailable or ambiguous; contact order is assumed "
            "to be FL, FR, RL, RR. Verify this once against the URDF."
        )

    all_rows: List[Dict[str, Any]] = []
    for source_gait, target_gait in transitions:
        source_matrix = gait_lookup[source_gait][1]
        target_matrix = gait_lookup[target_gait][1]
        for speed in map(float, args.speeds):
            condition_seed = stable_condition_seed(
                args.eval_seed, source_gait, target_gait, speed
            )
            random.seed(condition_seed)
            np.random.seed(condition_seed)
            torch.manual_seed(condition_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(condition_seed)

            env.reset_idx(all_ids)
            set_fixed_gait(env, source_gait)
            set_fixed_command(env, speed)
            cpg_rng = np.random.default_rng(condition_seed)
            cpg_parameters = set_cpg_episode_parameters(env, args, cpg_rng, torch)
            env.compute_observations()
            obs = env.get_observations()

            active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            fell_before = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            fell_after = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            fall_step_after = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            )

            # Establish the source gait.  No source-settle samples enter metrics.
            for _ in range(settle_steps):
                set_fixed_command(env, speed)
                with torch.no_grad():
                    actions = policy(obs.detach())
                    obs, _, _, dones, _ = env.step(actions.detach())
                dones = dones.bool()
                fell_before |= active & dones
                active &= ~dones

            pre_storage: Dict[str, List[np.ndarray]] = {}
            for _ in range(pre_steps):
                set_fixed_command(env, speed)
                with torch.no_grad():
                    actions = policy(obs.detach())
                    obs, _, _, dones, _ = env.step(actions.detach())
                dones = dones.bool()
                valid = active & (~dones)
                fell_before |= active & dones
                active &= ~dones
                append_trace(
                    pre_storage,
                    **record_state(
                        env=env,
                        actions=actions,
                        valid=valid,
                        source_matrix=source_matrix,
                        target_matrix=target_matrix,
                        foot_order=foot_order,
                        get_euler_xyz=get_euler_xyz,
                        contact_threshold_n=args.contact_threshold_n,
                        torch=torch,
                    ),
                )

            # The intervention: change only PHI_batch/current gait metadata.
            switch_gait_without_reset(env, target_gait)
            target_error_at_switch = phase_error(env, target_matrix, torch).detach().cpu().numpy()

            post_storage: Dict[str, List[np.ndarray]] = {}
            for post_step in range(post_steps):
                set_fixed_command(env, speed)
                with torch.no_grad():
                    actions = policy(obs.detach())
                    obs, _, _, dones, _ = env.step(actions.detach())
                dones = dones.bool()
                newly_fallen = active & dones
                fell_after |= newly_fallen
                unset = newly_fallen & (fall_step_after < 0)
                fall_step_after[unset] = post_step
                valid = active & (~dones)
                active &= ~dones
                append_trace(
                    post_storage,
                    **record_state(
                        env=env,
                        actions=actions,
                        valid=valid,
                        source_matrix=source_matrix,
                        target_matrix=target_matrix,
                        foot_order=foot_order,
                        get_euler_xyz=get_euler_xyz,
                        contact_threshold_n=args.contact_threshold_n,
                        torch=torch,
                    ),
                )

            pre_trace = stack_trace(pre_storage)
            post_trace = stack_trace(post_storage)
            fell_before_np = fell_before.detach().cpu().numpy()
            fell_after_np = fell_after.detach().cpu().numpy()
            fall_step_np = fall_step_after.detach().cpu().numpy()

            for episode in range(env.num_envs):
                metrics = analyze_episode(
                    pre_trace=pre_trace,
                    post_trace=post_trace,
                    episode=episode,
                    source_gait=source_gait,
                    target_gait=target_gait,
                    speed=speed,
                    gait_names=gait_names,
                    gait_offsets=gait_offsets,
                    args=args,
                    dt=dt,
                    leg_indices=leg_indices,
                    torque_limits=torque_limits,
                    spine_index=spine_index if has_active_spine else None,
                    robot_mass=robot_mass,
                    target_error_at_switch=float(target_error_at_switch[episode]),
                    fell_before_switch=bool(fell_before_np[episode]),
                    fell_after_switch=bool(fell_after_np[episode]),
                    fall_step_after_switch=int(fall_step_np[episode]),
                )
                all_rows.append(
                    {
                        "variant": variant,
                        "task": legged.task,
                        **control_info,
                        "source_gait": source_gait,
                        "target_gait": target_gait,
                        "transition": f"{source_gait}->{target_gait}",
                        "command_mps": speed,
                        "episode": episode,
                        "condition_seed": condition_seed,
                        "cpg_parameter_mode": args.cpg_parameter_mode,
                        "robot_height_m": float(cpg_parameters["robot_height_m"][episode]),
                        "ground_clearance_m": float(
                            cpg_parameters["ground_clearance_m"][episode]
                        ),
                        "ground_penetration_m": float(
                            cpg_parameters["ground_penetration_m"][episode]
                        ),
                        "offset_x_control": cpg_parameters["offset_x_control"],
                        "foot_order_source": foot_order_source,
                        **metrics,
                        "robot_mass_kg": robot_mass,
                    }
                )

            if args.save_timeseries:
                filename = (
                    f"{safe_name(source_gait)}_to_{safe_name(target_gait)}_"
                    f"{speed:.2f}mps.npz"
                )
                save_transition_timeseries(
                    variant_dir / "timeseries" / filename,
                    pre_trace,
                    post_trace,
                    dt,
                    source_gait,
                    target_gait,
                    speed,
                )

            rows = all_rows[-env.num_envs :]
            eligible = [row for row in rows if math.isfinite(float(row["transition_success"]))]
            success_rate = (
                statistics.fmean(float(row["transition_success"]) for row in eligible)
                if eligible
                else math.nan
            )
            contact_times = [
                float(row["contact_transition_time_s"])
                for row in rows
                if math.isfinite(float(row["contact_transition_time_s"]))
            ]
            mean_contact_time = statistics.fmean(contact_times) if contact_times else math.nan
            print(
                f"[{variant}] {source_gait:18s}->{target_gait:18s} "
                f"{speed:.2f} m/s eligible={len(eligible)}/{env.num_envs}, "
                f"success={success_rate:.3f}, contact_t={mean_contact_time:.3f} s"
            )

    write_csv(summary_path, all_rows, SUMMARY_FIELDS)
    metadata = {
        "variant": variant,
        "detected_variant": detected_variant,
        "task": legged.task,
        **control_info,
        "transitions": [f"{source}->{target}" for source, target in transitions],
        "speeds_mps": list(map(float, args.speeds)),
        "episodes_per_condition": args.episodes,
        "source_settle_s": args.source_settle_s,
        "pre_window_s": args.pre_window_s,
        "post_window_s": args.post_window_s,
        "final_window_s": args.final_window_s,
        "dt_s": dt,
        "eval_seed": args.eval_seed,
        "phase_threshold_deg": args.phase_threshold_deg,
        "phase_hold_s": args.phase_hold_s,
        "contact_score_threshold_deg": args.contact_score_threshold_deg,
        "contact_min_margin_deg": args.contact_min_margin_deg,
        "contact_match_fraction": args.contact_match_fraction,
        "contact_hold_s": args.contact_hold_s,
        "contact_lookback_s": args.contact_lookback_s,
        "contact_strikes_per_foot": args.contact_strikes_per_foot,
        "contact_min_strike_separation_s": args.contact_min_strike_separation_s,
        "velocity_abs_tolerance_mps": args.velocity_abs_tolerance_mps,
        "velocity_rel_tolerance": args.velocity_rel_tolerance,
        "velocity_hold_s": args.velocity_hold_s,
        "velocity_final_fraction": args.velocity_final_fraction,
        "domain_randomization_kept": args.keep_domain_rand,
        "cpg_parameter_mode": args.cpg_parameter_mode,
        "gait_template_names": gait_names,
        "gait_template_offsets_rad": gait_offsets.tolist(),
        "foot_order": list(LEG_NAMES),
        "foot_order_source": foot_order_source,
        "robot_mass_kg": robot_mass,
        "load_run": getattr(legged, "load_run", None),
        "checkpoint": getattr(legged, "checkpoint", None),
    }
    (variant_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(all_rows)} episode summaries to {summary_path}")


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_summary_files(input_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(input_dir.glob("**/summary.csv")):
        if path.name == "combined_summary.csv":
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def mean_sd_ci(values: Iterable[float]) -> Tuple[int, float, float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    count = int(finite.size)
    if count == 0:
        return 0, math.nan, math.nan, math.nan
    mean = float(np.mean(finite))
    if count == 1:
        return count, mean, math.nan, math.nan
    sd = float(np.std(finite, ddof=1))
    ci = 1.96 * sd / math.sqrt(count)
    return count, mean, sd, ci


def aggregate_rows(
    rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups: Dict[Tuple[str, str, str, float], List[Mapping[str, str]]] = {}
    for row in rows:
        key = (
            str(row["variant"]),
            str(row["source_gait"]),
            str(row["target_gait"]),
            to_float(row["command_mps"]),
        )
        groups.setdefault(key, []).append(row)

    fields = [
        "variant",
        "source_gait",
        "target_gait",
        "transition",
        "command_mps",
        "episodes",
    ]
    for metric in AGGREGATE_METRICS:
        fields.extend(
            [
                f"{metric}_n",
                f"{metric}_mean",
                f"{metric}_sd",
                f"{metric}_ci95",
            ]
        )

    output: List[Dict[str, Any]] = []
    for (variant, source, target, speed), group in sorted(groups.items()):
        record: Dict[str, Any] = {
            "variant": variant,
            "source_gait": source,
            "target_gait": target,
            "transition": f"{source}->{target}",
            "command_mps": speed,
            "episodes": len(group),
        }
        for metric in AGGREGATE_METRICS:
            count, mean, sd, ci = mean_sd_ci(to_float(row.get(metric)) for row in group)
            record[f"{metric}_n"] = count
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_ci95"] = ci
        output.append(record)
    return output, fields


def make_comparison_plots(
    aggregate: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV tables were written without plots.")
        return

    variants = sorted({str(row["variant"]) for row in aggregate})
    transitions = list(
        dict.fromkeys(
            (str(row["source_gait"]), str(row["target_gait"])) for row in aggregate
        )
    )

    def plot_metric(
        metric: str,
        ylabel: str,
        filename: str,
        ylim: Tuple[float, float] | None = None,
    ) -> None:
        if not transitions:
            return
        columns = min(3, len(transitions))
        rows_n = int(math.ceil(len(transitions) / columns))
        fig, axes = plt.subplots(
            rows_n,
            columns,
            figsize=(5.0 * columns, 3.7 * rows_n),
            squeeze=False,
            sharex=False,
        )
        for axis, (source, target) in zip(axes.flat, transitions):
            for variant in variants:
                points = [
                    row
                    for row in aggregate
                    if str(row["variant"]) == variant
                    and str(row["source_gait"]) == source
                    and str(row["target_gait"]) == target
                ]
                points.sort(key=lambda row: float(row["command_mps"]))
                x = np.asarray([float(row["command_mps"]) for row in points])
                y = np.asarray([to_float(row.get(f"{metric}_mean")) for row in points])
                ci = np.asarray([to_float(row.get(f"{metric}_ci95")) for row in points])
                finite = np.isfinite(x) & np.isfinite(y)
                if not np.any(finite):
                    continue
                yerr = np.where(np.isfinite(ci[finite]), ci[finite], 0.0)
                axis.errorbar(
                    x[finite], y[finite], yerr=yerr, marker="o", capsize=2, label=variant
                )
            axis.set_title(f"{source.replace('_', ' ').title()} → {target.replace('_', ' ').title()}")
            axis.set_xlabel("Commanded speed [m/s]")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            if ylim is not None:
                axis.set_ylim(*ylim)
        for axis in axes.flat[len(transitions) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(variants)))
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        else:
            fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    plot_metric(
        "transition_success", "Eligible transition success rate", "transition_success_rate.png", (0.0, 1.05)
    )
    plot_metric(
        "source_ready", "Source-ready fraction", "source_ready_rate.png", (0.0, 1.05)
    )
    plot_metric(
        "contact_transition_time_s", "Contact-gait recovery time [s]", "contact_transition_time.png"
    )
    plot_metric(
        "cpg_transition_time_s", "CPG phase-settling time [s]", "cpg_transition_time.png"
    )
    plot_metric(
        "velocity_recovery_time_s", "Velocity recovery time [s]", "velocity_recovery_time.png"
    )
    plot_metric(
        "peak_abs_velocity_error_mps", "Peak |velocity error| [m/s]", "peak_velocity_error.png"
    )
    plot_metric("peak_abs_pitch_deg", "Peak |pitch| [deg]", "peak_pitch.png")
    plot_metric(
        "requested_leg_torque_saturation_fraction",
        "Requested leg-torque saturation fraction",
        "requested_torque_saturation.png",
        (0.0, 1.0),
    )


def run_comparison(args: argparse.Namespace) -> None:
    input_dir = args.input_dir or args.output_dir
    output_dir = args.output_dir
    rows = read_summary_files(input_dir)
    if not rows:
        raise SystemExit(f"No variant summary.csv files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "combined_summary.csv", rows, SUMMARY_FIELDS)
    aggregate, fields = aggregate_rows(rows)
    write_csv(
        output_dir / "aggregate_by_variant_transition_speed.csv", aggregate, fields
    )
    make_comparison_plots(aggregate, output_dir)
    print(
        f"Combined {len(rows)} episode rows from "
        f"{len(set(row['variant'] for row in rows))} variants."
    )
    print(f"Saved comparison results to {output_dir}")


def main() -> None:
    args, remaining = parse_evaluation_args()
    if args.eval_mode == "collect":
        run_collection(args, remaining)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
