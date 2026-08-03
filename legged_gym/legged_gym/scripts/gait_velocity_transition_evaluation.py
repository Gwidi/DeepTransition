#!/usr/bin/env python3
"""Evaluate simultaneous gait and forward-velocity command steps.

This evaluator complements ``gait_transition_evaluation.py``.  It measures the
response when the CPG gait matrix and the forward-velocity command change at
the same simulation instant, and it collects matched velocity-only controls.
The primary comparison is therefore

    combined gait+velocity response - matched velocity-only response.

The default experiment contains six combined command steps and five unique
velocity-only controls.  With 30 episodes and three controller variants this
is 990 rollouts.  Adding ``gait_only`` to ``--interventions`` produces the
formally matched three-way design (1530 rollouts across three variants).

The script is intentionally strict about the validated policy action layouts:

* rigid spine: 12 actions (4 amplitudes, 4 frequencies, 4 foot x offsets);
* direct/phase-locked spine: 13 actions; and
* uncoupled spine: 14 actions.

It rejects the older 8/10-action configuration, because those policies do not
contain the four per-foot x-offset actions used by the validated results.

Evaluator-specific arguments are parsed here.  Unknown arguments, including
``--task``, ``--load_run``, ``--checkpoint``, and ``--headless``, are passed to
``legged_gym.utils.get_args``.

Examples
--------
Collect one controller variant::

    python gait_velocity_transition_evaluation.py \
      --eval-mode collect --variant rigid \
      --output-dir gait_velocity_transition_results \
      --task <registered-task> --load_run <run> --checkpoint <checkpoint> \
      --headless --save-timeseries

Combine all collected variants and calculate paired effects::

    python gait_velocity_transition_evaluation.py \
      --eval-mode compare \
      --input-dir gait_velocity_transition_results \
      --output-dir gait_velocity_transition_results/comparison \
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import gait_transition_evaluation as gait_eval
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
    write_csv,
)


DEFAULT_COMMAND_STEPS = (
    "WALK:0.6:TROT:1.2",
    "TROT:1.2:WALK:0.6",
    "TROT:1.2:BOUND:1.8",
    "BOUND:1.8:TROT:1.2",
    "TROT:1.2:ROTARY_GALLOP:1.8",
    "ROTARY_GALLOP:1.8:TROT:1.2",
)
DEFAULT_INTERVENTIONS = ("combined", "velocity_only")
INTERVENTION_CHOICES = ("combined", "velocity_only", "gait_only")
LEG_NAMES = tuple(gait_eval.LEG_NAMES)

WINDOWS: Tuple[Tuple[str, float | None], ...] = (
    ("1s", 1.0),
    ("2s", 2.0),
    ("post", None),
)

WINDOW_METRIC_STEMS = (
    "velocity_error_iae_{label}_m",
    "velocity_within_tolerance_fraction_{label}",
    "peak_abs_pitch_change_{label}_deg",
    "pitch_change_rms_{label}_deg",
    "peak_abs_roll_change_{label}_deg",
    "peak_foot_contact_force_{label}_n",
    "leg_torque_saturation_fraction_{label}",
    "requested_leg_torque_saturation_fraction_{label}",
    "requested_torque_peak_ratio_{label}",
    "requested_torque_exceedance_mean_{label}_nm",
    "longest_requested_saturation_{label}_s",
    "leg_joint_tracking_rmse_{label}_rad",
    "total_gross_work_{label}_j",
    "leg_gross_work_{label}_j",
    "spine_gross_work_{label}_j",
    "forward_distance_{label}_m",
    "work_per_forward_m_{label}_jpm",
    "cost_of_transport_{label}",
)


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
    "action_layout_name",
    "offset_action_start",
    "offset_action_stop",
    "offset_x_from_policy_actions",
    "intervention",
    "condition_id",
    "comparison_ids",
    "source_gait",
    "target_gait",
    "source_speed_mps",
    "target_speed_mps",
    "velocity_step_mps",
    "gait_command_changed",
    "velocity_command_changed",
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
    "source_phase_error_final_deg",
    "source_contact_realized_gait",
    "source_contact_match_fraction",
    "source_contact_score_deg",
    "source_contact_margin_deg",
    "pre_mean_velocity_mps",
    "pre_velocity_rmse_mps",
    "pre_velocity_final_within_tolerance_fraction",
    "pre_mean_pitch_deg",
    "cpg_target_error_at_switch_deg",
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
    "final_velocity_bias_mps",
    "post_velocity_rmse_mps",
    "normalized_step_response_2s",
    "normalized_step_response_final",
    "velocity_step_completion_fraction_2s",
    "velocity_step_completion_fraction_final",
    "velocity_overshoot_fraction",
    "velocity_overshoot_pct",
]
for _label, _seconds in WINDOWS:
    SUMMARY_FIELDS.extend(stem.format(label=_label) for stem in WINDOW_METRIC_STEMS)
SUMMARY_FIELDS.extend(
    [
        "post_spine_angle_rms_rad",
        "post_spine_angle_range_rad",
        "post_offset_x_fl_mean_m",
        "post_offset_x_fr_mean_m",
        "post_offset_x_rl_mean_m",
        "post_offset_x_rr_mean_m",
        "transition_success",
        "robot_mass_kg",
    ]
)


AGGREGATE_METRICS = [
    "fell_after_switch",
    "source_ready",
    "cpg_target_error_final_deg",
    "cpg_transition_completed",
    "cpg_transition_time_s",
    "contact_transition_completed",
    "contact_transition_time_s",
    "target_contact_stable",
    "target_contact_match_fraction",
    "velocity_recovered",
    "velocity_recovery_time_s",
    "velocity_final_stable",
    "post_velocity_bias_mps",
    "final_velocity_bias_mps",
    "velocity_step_completion_fraction_2s",
    "velocity_step_completion_fraction_final",
    "velocity_overshoot_pct",
    "velocity_error_iae_2s_m",
    "velocity_error_iae_post_m",
    "peak_abs_pitch_change_2s_deg",
    "peak_abs_pitch_change_post_deg",
    "peak_foot_contact_force_2s_n",
    "peak_foot_contact_force_post_n",
    "requested_leg_torque_saturation_fraction_2s",
    "requested_leg_torque_saturation_fraction_post",
    "total_gross_work_2s_j",
    "total_gross_work_post_j",
    "work_per_forward_m_2s_jpm",
    "work_per_forward_m_post_jpm",
    "cost_of_transport_2s",
    "cost_of_transport_post",
    "transition_success",
]


PAIRED_EFFECT_METRICS = [
    "fell_after_switch",
    "velocity_error_iae_2s_m",
    "velocity_error_iae_post_m",
    "velocity_step_completion_fraction_2s",
    "velocity_step_completion_fraction_final",
    "velocity_overshoot_pct",
    "peak_abs_pitch_change_2s_deg",
    "peak_abs_pitch_change_post_deg",
    "peak_foot_contact_force_2s_n",
    "requested_leg_torque_saturation_fraction_2s",
    "total_gross_work_2s_j",
    "leg_gross_work_2s_j",
    "spine_gross_work_2s_j",
    "work_per_forward_m_2s_jpm",
    "cost_of_transport_2s",
]


@dataclass(frozen=True)
class RequestedCommandStep:
    source_gait: str
    source_speed_mps: float
    target_gait: str
    target_speed_mps: float

    @property
    def comparison_id(self) -> str:
        return (
            f"{safe_name(self.source_gait)}_{self.source_speed_mps:.2f}_to_"
            f"{safe_name(self.target_gait)}_{self.target_speed_mps:.2f}"
        )


@dataclass(frozen=True)
class TrialCondition:
    intervention: str
    source_gait: str
    source_speed_mps: float
    target_gait: str
    target_speed_mps: float
    comparison_ids: Tuple[str, ...]
    seed_target_speed_mps: float

    @property
    def condition_id(self) -> str:
        return (
            f"{self.intervention}__{safe_name(self.source_gait)}_"
            f"{self.source_speed_mps:.2f}_to_{safe_name(self.target_gait)}_"
            f"{self.target_speed_mps:.2f}"
        )


def normalize_gait(value: str) -> str:
    return value.strip().upper().replace("-", "_").replace(" ", "_")


def parse_command_step(value: str) -> RequestedCommandStep:
    normalized = value.strip().replace("->", ":").replace("@", ":").replace(",", ":")
    parts = [part.strip() for part in normalized.split(":") if part.strip()]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid command step {value!r}; use SOURCE_GAIT:SOURCE_SPEED:"
            "TARGET_GAIT:TARGET_SPEED (for example WALK:0.6:TROT:1.2)."
        )
    source_gait = normalize_gait(parts[0])
    target_gait = normalize_gait(parts[2])
    try:
        source_speed = float(parts[1])
        target_speed = float(parts[3])
    except ValueError as exc:
        raise ValueError(f"Invalid speed in command step {value!r}") from exc
    if source_speed <= 0.0 or target_speed <= 0.0:
        raise ValueError(f"Command-step speeds must be positive: {value!r}")
    if source_gait == target_gait:
        raise ValueError(
            f"A combined command step must change gait; got {value!r}. "
            "Velocity-only controls are generated automatically."
        )
    if math.isclose(source_speed, target_speed, abs_tol=1.0e-12):
        raise ValueError(
            f"A combined command step must change speed; got {value!r}. "
            "Use gait_transition_evaluation.py for constant-speed switches."
        )
    return RequestedCommandStep(source_gait, source_speed, target_gait, target_speed)


def requested_steps(args: argparse.Namespace) -> List[RequestedCommandStep]:
    parsed = [parse_command_step(value) for value in args.command_steps]
    return list(dict.fromkeys(parsed))


def build_trial_conditions(
    steps: Sequence[RequestedCommandStep],
    interventions: Sequence[str],
) -> List[TrialCondition]:
    output: List[TrialCondition] = []
    interventions = tuple(dict.fromkeys(interventions))
    if "combined" in interventions:
        for step in steps:
            output.append(
                TrialCondition(
                    intervention="combined",
                    source_gait=step.source_gait,
                    source_speed_mps=step.source_speed_mps,
                    target_gait=step.target_gait,
                    target_speed_mps=step.target_speed_mps,
                    comparison_ids=(step.comparison_id,),
                    seed_target_speed_mps=step.target_speed_mps,
                )
            )

    if "velocity_only" in interventions:
        grouped: Dict[Tuple[str, float, float], List[str]] = {}
        for step in steps:
            key = (step.source_gait, step.source_speed_mps, step.target_speed_mps)
            grouped.setdefault(key, []).append(step.comparison_id)
        for (source_gait, source_speed, target_speed), comparison_ids in grouped.items():
            output.append(
                TrialCondition(
                    intervention="velocity_only",
                    source_gait=source_gait,
                    source_speed_mps=source_speed,
                    target_gait=source_gait,
                    target_speed_mps=target_speed,
                    comparison_ids=tuple(comparison_ids),
                    seed_target_speed_mps=target_speed,
                )
            )

    if "gait_only" in interventions:
        for step in steps:
            output.append(
                TrialCondition(
                    intervention="gait_only",
                    source_gait=step.source_gait,
                    source_speed_mps=step.source_speed_mps,
                    target_gait=step.target_gait,
                    target_speed_mps=step.source_speed_mps,
                    comparison_ids=(step.comparison_id,),
                    seed_target_speed_mps=step.target_speed_mps,
                )
            )
    return output


def parse_evaluation_args(
    argv: Sequence[str] | None = None,
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-mode", choices=("collect", "compare"), default="collect")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("gait_velocity_transition_results")
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--command-steps", nargs="+", default=list(DEFAULT_COMMAND_STEPS)
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=INTERVENTION_CHOICES,
        default=list(DEFAULT_INTERVENTIONS),
    )
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
        help="Use training-range midpoints or paired deterministic episode samples.",
    )
    parser.add_argument("--robot-height-m", type=float, default=None)
    parser.add_argument("--ground-clearance-m", type=float, default=None)
    parser.add_argument("--ground-penetration-m", type=float, default=None)
    parser.add_argument("--offset-x-m", type=float, default=None)
    parser.add_argument("--save-timeseries", action="store_true")
    parser.add_argument("--keep-domain-rand", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gait-velocity-help", action="store_true")
    args, remaining = parser.parse_known_args(argv)

    if args.gait_velocity_help:
        print(__doc__)
        print("Evaluator options:\n")
        parser.print_help()
        print("\nAll remaining options are passed to legged_gym.utils.get_args().")
        raise SystemExit(0)
    return args, remaining


def validate_args(args: argparse.Namespace) -> List[RequestedCommandStep]:
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    for name in ("source_settle_s", "pre_window_s", "post_window_s", "final_window_s"):
        if float(getattr(args, name)) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.post_window_s < 2.0:
        raise SystemExit("--post-window-s must be at least 2.0 s")
    if args.contact_strikes_per_foot < 1:
        raise SystemExit("--contact-strikes-per-foot must be at least 1")
    for name in ("contact_match_fraction", "velocity_final_fraction"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be at least 1")
    try:
        return requested_steps(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def configure_evaluation_environment(
    env_cfg: Any,
    args: argparse.Namespace,
    steps: Sequence[RequestedCommandStep],
) -> None:
    total_s = args.source_settle_s + args.pre_window_s + args.post_window_s
    env_cfg.env.num_envs = args.episodes
    set_if_present(env_cfg.env, "play", True)
    env_cfg.env.episode_length_s = max(
        float(env_cfg.env.episode_length_s), total_s + 2.0
    )

    set_if_present(env_cfg.commands, "curriculum", False)
    set_if_present(env_cfg.commands, "resample_gait_style", False)
    set_if_present(env_cfg.commands, "resampling_time", 1.0e9)
    set_if_present(env_cfg.commands, "gait_resampling_time", 1.0e9)
    if hasattr(env_cfg.commands, "ranges"):
        speeds = [
            value
            for step in steps
            for value in (step.source_speed_mps, step.target_speed_mps)
        ]
        set_if_present(
            env_cfg.commands.ranges, "lin_vel_x", [min(speeds), max(speeds)]
        )
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


def stable_condition_seed(
    base_seed: int,
    source_gait: str,
    source_speed_mps: float,
    target_speed_mps: float,
) -> int:
    # Target gait and intervention are deliberately absent.  A combined trial
    # and its velocity-only control therefore receive identical environment and
    # per-episode CPG random samples before the command step.
    payload = (
        f"{base_seed}|{source_gait}|{source_speed_mps:.8f}|"
        f"{target_speed_mps:.8f}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**31 - 1))


def validated_action_layout(env: Any) -> Dict[str, Any]:
    control = env.cfg.control
    control_type = str(getattr(control, "control_type", "")).upper()
    offset_from_actions = bool(getattr(env._cpg, "_offset_x_from_actions", False))
    if "OFFSETX_ACTION" not in control_type or not offset_from_actions:
        raise RuntimeError(
            "This evaluator requires the validated OFFSETX_ACTION policy revision. "
            f"Loaded control_type={control_type!r}, "
            f"_offset_x_from_actions={offset_from_actions}. The older 8/10-action "
            "configuration is not compatible."
        )

    spine_locked = bool(getattr(env.cfg.asset, "spine_locked", False))
    control_mode = str(getattr(control, "spine_control_mode", "cpg")).lower()
    phase_mode = str(getattr(control, "spine_phase_mode", "phase_locked")).lower()
    if spine_locked:
        layout_name = "rigid_offsetx"
        expected_actions = 12
        offset_start = 8
    elif control_mode == "direct":
        layout_name = "direct_spine_offsetx"
        expected_actions = 13
        offset_start = 9
    elif phase_mode == "uncoupled":
        layout_name = "uncoupled_spine_offsetx"
        expected_actions = 14
        offset_start = 10
    else:
        layout_name = "phase_locked_spine_offsetx"
        expected_actions = 13
        offset_start = 9

    actual_actions = int(env.num_actions)
    if actual_actions != expected_actions:
        raise RuntimeError(
            f"Validated action-layout check failed for {layout_name}: expected "
            f"{expected_actions} actions, got {actual_actions}. Do not run this "
            "experiment with a checkpoint/config pair from a different revision."
        )
    return {
        "action_layout_name": layout_name,
        "expected_num_actions": expected_actions,
        "offset_action_start": offset_start,
        "offset_action_stop": offset_start + 4,
        "offset_x_from_policy_actions": True,
    }


def assert_policy_action_shape(actions: Any, env: Any, layout: Mapping[str, Any]) -> None:
    expected = (int(env.num_envs), int(layout["expected_num_actions"]))
    actual = tuple(int(value) for value in actions.shape)
    if actual != expected:
        raise RuntimeError(
            f"Policy returned actions with shape {actual}; expected {expected}. "
            "Check that --load_run and --checkpoint belong to the registered task."
        )


def first_window_mask(valid: np.ndarray, seconds: float | None, dt: float) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    if seconds is None:
        return valid.copy()
    steps = min(valid.size, max(1, int(round(float(seconds) / dt))))
    mask = np.zeros_like(valid)
    mask[:steps] = valid[:steps]
    return mask


def ending_window_mask(
    valid: np.ndarray,
    end_s: float | None,
    width_s: float,
    dt: float,
) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    end = valid.size if end_s is None else min(valid.size, max(1, int(round(end_s / dt))))
    width = max(1, int(round(width_s / dt)))
    start = max(0, end - width)
    mask = np.zeros_like(valid)
    mask[start:end] = valid[start:end]
    return mask


def safe_fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return math.nan
    return float(np.mean(np.asarray(mask, dtype=bool)[valid]))


def safe_circular_mean(values: np.ndarray, valid: np.ndarray) -> float:
    selected = gait_eval.finite_values(values, valid)
    return gait_eval.circular_mean(selected) if selected.size else math.nan


def circular_delta(values: np.ndarray, reference: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.arctan2(np.sin(values - reference), np.cos(values - reference))


def finite_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return numerator / denominator


def compute_window_metrics(
    post: Mapping[str, np.ndarray],
    mask: np.ndarray,
    label: str,
    target_speed_mps: float,
    velocity_tolerance_mps: float,
    pre_pitch_rad: float,
    pre_roll_rad: float,
    dt: float,
    leg_indices: np.ndarray,
    torque_limits: np.ndarray,
    spine_index: int | None,
    robot_mass_kg: float,
) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    velocity = np.asarray(post["velocity_mps"], dtype=np.float64)
    velocity_error = velocity - float(target_speed_mps)
    velocity_ok = np.isfinite(velocity_error) & (np.abs(velocity_error) <= velocity_tolerance_mps)

    pitch_delta = circular_delta(post["pitch_rad"], pre_pitch_rad)
    roll_delta = circular_delta(post["roll_rad"], pre_roll_rad)
    foot_forces = np.asarray(post["foot_contact_force_n"], dtype=np.float64)

    applied = np.asarray(post["applied_torque_nm"], dtype=np.float64)
    applied_leg = applied[:, leg_indices]
    leg_limits = np.asarray(torque_limits, dtype=np.float64)[leg_indices]
    applied_saturation = np.abs(applied_leg) >= (leg_limits[None, :] - 1.0e-5)
    applied_fraction = float(np.mean(applied_saturation[mask])) if np.any(mask) else math.nan

    requested_leg = None
    if "requested_torque_nm" in post:
        requested_leg = np.asarray(post["requested_torque_nm"], dtype=np.float64)[:, leg_indices]
    requested_fraction, peak_ratio, exceedance, longest = gait_eval._requested_saturation_metrics(
        requested_leg, leg_limits, mask, dt
    )

    dof_velocity = np.asarray(post["dof_velocity_rads"], dtype=np.float64)
    joint_power = np.abs(applied * dof_velocity)
    total_work = float(np.sum(joint_power[mask]) * dt) if np.any(mask) else math.nan
    leg_work = (
        float(np.sum(joint_power[mask][:, leg_indices]) * dt) if np.any(mask) else math.nan
    )
    spine_work = (
        float(np.sum(joint_power[mask][:, spine_index]) * dt)
        if spine_index is not None and np.any(mask)
        else math.nan
    )
    distance = (
        float(np.sum(np.maximum(velocity[mask], 0.0)) * dt) if np.any(mask) else math.nan
    )
    work_per_distance = finite_divide(total_work, distance)
    cot_denominator = robot_mass_kg * 9.81 * distance
    cot = finite_divide(total_work, cot_denominator)

    joint_tracking_rmse = math.nan
    if "dof_desired_position_rad" in post:
        tracking_error = (
            np.asarray(post["dof_desired_position_rad"], dtype=np.float64)[:, leg_indices]
            - np.asarray(post["dof_position_rad"], dtype=np.float64)[:, leg_indices]
        )
        joint_tracking_rmse = gait_eval.safe_rms(tracking_error, mask)

    force_values = gait_eval.finite_values(foot_forces, mask)
    peak_force = float(np.max(force_values)) if force_values.size else math.nan
    velocity_values = gait_eval.finite_values(np.abs(velocity_error), mask)
    velocity_iae = float(np.sum(velocity_values) * dt) if velocity_values.size else math.nan

    return {
        f"velocity_error_iae_{label}_m": velocity_iae,
        f"velocity_within_tolerance_fraction_{label}": safe_fraction(velocity_ok, mask),
        f"peak_abs_pitch_change_{label}_deg": math.degrees(
            gait_eval.safe_peak_abs(pitch_delta, mask)
        ),
        f"pitch_change_rms_{label}_deg": math.degrees(gait_eval.safe_rms(pitch_delta, mask)),
        f"peak_abs_roll_change_{label}_deg": math.degrees(
            gait_eval.safe_peak_abs(roll_delta, mask)
        ),
        f"peak_foot_contact_force_{label}_n": peak_force,
        f"leg_torque_saturation_fraction_{label}": applied_fraction,
        f"requested_leg_torque_saturation_fraction_{label}": requested_fraction,
        f"requested_torque_peak_ratio_{label}": peak_ratio,
        f"requested_torque_exceedance_mean_{label}_nm": exceedance,
        f"longest_requested_saturation_{label}_s": longest,
        f"leg_joint_tracking_rmse_{label}_rad": joint_tracking_rmse,
        f"total_gross_work_{label}_j": total_work,
        f"leg_gross_work_{label}_j": leg_work,
        f"spine_gross_work_{label}_j": spine_work,
        f"forward_distance_{label}_m": distance,
        f"work_per_forward_m_{label}_jpm": work_per_distance,
        f"cost_of_transport_{label}": cot,
    }


def analyze_episode(
    pre_trace: Mapping[str, np.ndarray],
    post_trace: Mapping[str, np.ndarray],
    episode: int,
    condition: TrialCondition,
    gait_names: Sequence[str],
    gait_offsets: np.ndarray,
    args: argparse.Namespace,
    dt: float,
    leg_indices: np.ndarray,
    torque_limits: np.ndarray,
    spine_index: int | None,
    robot_mass_kg: float,
    target_error_at_switch: float,
    fell_before_switch: bool,
    fell_after_switch: bool,
    fall_step_after_switch: int,
) -> Dict[str, Any]:
    pre = gait_eval._episode_trace(pre_trace, episode)
    post = gait_eval._episode_trace(post_trace, episode)
    pre_valid = np.asarray(pre["valid"], dtype=bool)
    post_valid = np.asarray(post["valid"], dtype=bool)
    source_index = list(gait_names).index(condition.source_gait)
    target_index = list(gait_names).index(condition.target_gait)
    final_steps = max(1, int(round(args.final_window_s / dt)))
    phase_hold_steps = max(1, int(round(args.phase_hold_s / dt)))
    contact_hold_steps = max(1, int(round(args.contact_hold_s / dt)))
    velocity_hold_steps = max(1, int(round(args.velocity_hold_s / dt)))
    lookback_steps = max(1, int(round(args.contact_lookback_s / dt)))
    separation_steps = max(1, int(round(args.contact_min_strike_separation_s / dt)))

    pre_classification = gait_eval.touchdown_classification(
        reference_phase=pre["cpg_phase_rad"][:, 0],
        contacts=pre["contacts"],
        valid=pre_valid,
        gait_offsets=gait_offsets,
        lookback_steps=lookback_steps,
        strikes_per_foot=args.contact_strikes_per_foot,
        min_separation_steps=separation_steps,
    )
    initial_contacts = pre["contacts"][-1] if pre["contacts"].shape[0] else None
    post_classification = gait_eval.touchdown_classification(
        reference_phase=post["cpg_phase_rad"][:, 0],
        contacts=post["contacts"],
        valid=post_valid,
        gait_offsets=gait_offsets,
        lookback_steps=lookback_steps,
        strikes_per_foot=args.contact_strikes_per_foot,
        min_separation_steps=separation_steps,
        initial_contacts=initial_contacts,
    )

    source_label, source_fraction, source_score, source_margin = (
        gait_eval.dominant_classification(
            pre_classification, source_index, pre_valid, final_steps
        )
    )
    target_label, target_fraction, target_score, target_margin = (
        gait_eval.dominant_classification(
            post_classification, target_index, post_valid, final_steps
        )
    )

    phase_threshold = math.radians(args.phase_threshold_deg)
    contact_threshold = math.radians(args.contact_score_threshold_deg)
    contact_margin_threshold = math.radians(args.contact_min_margin_deg)
    source_tolerance = max(
        float(args.velocity_abs_tolerance_mps),
        abs(condition.source_speed_mps) * float(args.velocity_rel_tolerance),
    )
    target_tolerance = max(
        float(args.velocity_abs_tolerance_mps),
        abs(condition.target_speed_mps) * float(args.velocity_rel_tolerance),
    )

    source_phase_final = gait_eval.safe_final_mean(
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
    pre_velocity = np.asarray(pre["velocity_mps"], dtype=np.float64)
    pre_velocity_error = pre_velocity - condition.source_speed_mps
    pre_velocity_ok = (
        pre_valid
        & np.isfinite(pre_velocity_error)
        & (np.abs(pre_velocity_error) <= source_tolerance)
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
    source_ready = source_contact_ready and source_velocity_ready

    target_phase_error = np.asarray(post["target_phase_error_rad"], dtype=np.float64)
    phase_ok = (
        post_valid
        & np.isfinite(target_phase_error)
        & (target_phase_error <= phase_threshold)
    )
    phase_index = gait_eval.first_sustained_true(phase_ok, phase_hold_steps)
    cpg_completed = phase_index is not None
    cpg_time = (phase_index + 1) * dt if phase_index is not None else math.nan

    best = post_classification["best_index"]
    target_scores = post_classification["scores_rad"][:, target_index]
    target_margins = post_classification["margin_rad"]
    contact_ok = (
        post_valid
        & (best == target_index)
        & np.isfinite(target_scores)
        & (target_scores <= contact_threshold)
        & np.isfinite(target_margins)
        & (target_margins >= contact_margin_threshold)
    )
    contact_index = gait_eval.first_sustained_true(contact_ok, contact_hold_steps)
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

    post_velocity = np.asarray(post["velocity_mps"], dtype=np.float64)
    velocity_error = post_velocity - condition.target_speed_mps
    velocity_ok = (
        post_valid
        & np.isfinite(velocity_error)
        & (np.abs(velocity_error) <= target_tolerance)
    )
    velocity_index = gait_eval.first_sustained_true(velocity_ok, velocity_hold_steps)
    velocity_recovered = velocity_index is not None
    velocity_time = (velocity_index + 1) * dt if velocity_index is not None else math.nan
    final_valid = post_valid[-final_steps:]
    final_velocity_fraction = (
        float(np.mean(velocity_ok[-final_steps:][final_valid]))
        if np.any(final_valid)
        else math.nan
    )
    final_stable = bool(
        math.isfinite(final_velocity_fraction)
        and final_velocity_fraction >= args.velocity_final_fraction
    )

    pre_mean_velocity = gait_eval.safe_mean(pre_velocity, pre_valid)
    step_denominator = condition.target_speed_mps - condition.source_speed_mps
    if math.isfinite(pre_mean_velocity) and not math.isclose(
        step_denominator, 0.0, abs_tol=1.0e-12
    ):
        normalized_response = (post_velocity - pre_mean_velocity) / step_denominator
        response_2s_mask = ending_window_mask(
            post_valid, 2.0, min(args.final_window_s, 2.0), dt
        )
        response_final_mask = ending_window_mask(
            post_valid, None, args.final_window_s, dt
        )
        response_2s = gait_eval.safe_mean(normalized_response, response_2s_mask)
        response_final = gait_eval.safe_mean(normalized_response, response_final_mask)
        response_values = gait_eval.finite_values(normalized_response, post_valid)
        overshoot = (
            max(0.0, float(np.max(response_values)) - 1.0)
            if response_values.size
            else math.nan
        )
    else:
        response_2s = math.nan
        response_final = math.nan
        overshoot = math.nan

    completion_2s = (
        float(np.clip(response_2s, 0.0, 1.0))
        if math.isfinite(response_2s)
        else math.nan
    )
    completion_final = (
        float(np.clip(response_final, 0.0, 1.0)) if math.isfinite(response_final) else math.nan
    )

    pre_pitch = safe_circular_mean(pre["pitch_rad"], pre_valid)
    pre_roll = safe_circular_mean(pre["roll_rad"], pre_valid)
    window_metrics: Dict[str, float] = {}
    for label, seconds in WINDOWS:
        window_metrics.update(
            compute_window_metrics(
                post=post,
                mask=first_window_mask(post_valid, seconds, dt),
                label=label,
                target_speed_mps=condition.target_speed_mps,
                velocity_tolerance_mps=target_tolerance,
                pre_pitch_rad=pre_pitch,
                pre_roll_rad=pre_roll,
                dt=dt,
                leg_indices=leg_indices,
                torque_limits=torque_limits,
                spine_index=spine_index,
                robot_mass_kg=robot_mass_kg,
            )
        )

    spine_rms = math.nan
    spine_range = math.nan
    if spine_index is not None and "spine_angle_rad" in post:
        spine_values = gait_eval.finite_values(post["spine_angle_rad"], post_valid)
        if spine_values.size:
            spine_rms = float(np.sqrt(np.mean(np.square(spine_values))))
            spine_range = float(np.max(spine_values) - np.min(spine_values))

    offset_means = np.full(4, np.nan, dtype=np.float64)
    if "offset_x_m" in post:
        offsets = np.asarray(post["offset_x_m"], dtype=np.float64)
        if offsets.ndim == 2 and offsets.shape[1] >= 4 and np.any(post_valid):
            offset_means = np.mean(offsets[post_valid, :4], axis=0)

    target_final_phase = gait_eval.safe_final_mean(target_phase_error, post_valid, final_steps)
    phase_values = gait_eval.finite_values(target_phase_error, post_valid)
    phase_iae = float(np.sum(phase_values) * dt) if phase_values.size else math.nan
    touchdowns = np.sum(post_classification["touchdown"] & post_valid[:, None], axis=0)
    source_name = gait_names[source_label] if source_label >= 0 else "UNAVAILABLE"
    target_name = gait_names[target_label] if target_label >= 0 else "UNAVAILABLE"
    post_time = float(np.sum(post_valid) * dt)
    fall_time = (
        float((fall_step_after_switch + 1) * dt)
        if fell_after_switch and fall_step_after_switch >= 0
        else math.nan
    )
    transition_success = (
        math.nan
        if not source_ready
        else float(
            (not fell_after_switch)
            and contact_completed
            and target_contact_stable
            and velocity_recovered
            and final_stable
        )
    )

    final_velocity_bias = (
        gait_eval.safe_mean(
            post_velocity,
            ending_window_mask(post_valid, None, args.final_window_s, dt),
        )
        - condition.target_speed_mps
    )
    output: Dict[str, Any] = {
        "fell_before_switch": int(fell_before_switch),
        "fell_after_switch": int(fell_after_switch),
        "fall_time_after_switch_s": fall_time,
        "valid_post_time_s": post_time,
        "source_cpg_ready": int(source_cpg_ready),
        "source_contact_ready": int(source_contact_ready),
        "source_velocity_ready": int(source_velocity_ready),
        "source_ready": int(source_ready),
        "source_phase_error_final_deg": math.degrees(source_phase_final),
        "source_contact_realized_gait": source_name,
        "source_contact_match_fraction": source_fraction,
        "source_contact_score_deg": math.degrees(source_score),
        "source_contact_margin_deg": math.degrees(source_margin),
        "pre_mean_velocity_mps": pre_mean_velocity,
        "pre_velocity_rmse_mps": gait_eval.safe_rms(pre_velocity_error, pre_valid),
        "pre_velocity_final_within_tolerance_fraction": pre_velocity_final_fraction,
        "pre_mean_pitch_deg": math.degrees(pre_pitch),
        "cpg_target_error_at_switch_deg": math.degrees(target_error_at_switch),
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
        "velocity_recovery_tolerance_mps": target_tolerance,
        "velocity_final_stable": int(final_stable),
        "velocity_final_within_tolerance_fraction": final_velocity_fraction,
        "post_mean_velocity_mps": gait_eval.safe_mean(post_velocity, post_valid),
        "post_velocity_bias_mps": gait_eval.safe_mean(post_velocity, post_valid)
        - condition.target_speed_mps,
        "final_velocity_bias_mps": final_velocity_bias,
        "post_velocity_rmse_mps": gait_eval.safe_rms(velocity_error, post_valid),
        "normalized_step_response_2s": response_2s,
        "normalized_step_response_final": response_final,
        "velocity_step_completion_fraction_2s": completion_2s,
        "velocity_step_completion_fraction_final": completion_final,
        "velocity_overshoot_fraction": overshoot,
        "velocity_overshoot_pct": overshoot * 100.0 if math.isfinite(overshoot) else math.nan,
        **window_metrics,
        "post_spine_angle_rms_rad": spine_rms,
        "post_spine_angle_range_rad": spine_range,
        "post_offset_x_fl_mean_m": float(offset_means[0]),
        "post_offset_x_fr_mean_m": float(offset_means[1]),
        "post_offset_x_rl_mean_m": float(offset_means[2]),
        "post_offset_x_rr_mean_m": float(offset_means[3]),
        "transition_success": transition_success,
    }
    return output


def save_timeseries(
    path: Path,
    pre_trace: Mapping[str, np.ndarray],
    post_trace: Mapping[str, np.ndarray],
    dt: float,
    condition: TrialCondition,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {}
    for name in sorted(set(pre_trace) & set(post_trace)):
        arrays[name] = np.concatenate((pre_trace[name], post_trace[name]), axis=0)
    pre_steps = next(iter(pre_trace.values())).shape[0]
    post_steps = next(iter(post_trace.values())).shape[0]
    arrays["time_s"] = np.concatenate(
        (
            np.arange(-pre_steps, 0, dtype=np.float64) * dt,
            np.arange(1, post_steps + 1, dtype=np.float64) * dt,
        )
    )
    arrays["command_mps"] = np.concatenate(
        (
            np.full(pre_steps, condition.source_speed_mps, dtype=np.float64),
            np.full(post_steps, condition.target_speed_mps, dtype=np.float64),
        )
    )
    arrays["transition_index"] = np.asarray(pre_steps, dtype=np.int64)
    arrays["intervention"] = np.asarray(condition.intervention)
    arrays["condition_id"] = np.asarray(condition.condition_id)
    arrays["comparison_ids"] = np.asarray(";".join(condition.comparison_ids))
    arrays["source_gait"] = np.asarray(condition.source_gait)
    arrays["target_gait"] = np.asarray(condition.target_gait)
    arrays["source_speed_mps"] = np.asarray(condition.source_speed_mps, dtype=np.float64)
    arrays["target_speed_mps"] = np.asarray(condition.target_speed_mps, dtype=np.float64)
    np.savez_compressed(path, **arrays)


def run_collection(args: argparse.Namespace, legged_args: Sequence[str]) -> None:
    steps = validate_args(args)
    conditions = build_trial_conditions(steps, args.interventions)
    if not conditions:
        raise SystemExit("No trial conditions were generated")

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
    configure_evaluation_environment(env_cfg, args, steps)
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
    try:
        action_layout = validated_action_layout(env)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    variant_dir = args.output_dir / safe_name(variant)
    summary_path = variant_dir / "summary.csv"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"{summary_path} already exists; pass --overwrite to replace it")
    variant_dir.mkdir(parents=True, exist_ok=True)

    gait_lookup = gait_eval.available_gaits(env)
    missing = sorted(
        {
            gait
            for condition in conditions
            for gait in (condition.source_gait, condition.target_gait)
            if gait not in gait_lookup
        }
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
    foot_order, foot_order_source = gait_eval.canonical_foot_order(env)
    gait_names, gait_offsets = gait_eval.gait_template_offsets(env)
    control_info = control_metadata(env)

    if gait_eval.get_requested_torques(env) is None:
        print(
            "Note: pre-clipping torque is not exposed by this environment. "
            "Requested-torque metrics will be NaN."
        )
    if foot_order_source.startswith("assumed"):
        print(
            "Note: feet_names was unavailable or ambiguous; contact order is assumed "
            "to be FL, FR, RL, RR. Verify this once against the URDF."
        )

    all_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        source_matrix = gait_lookup[condition.source_gait][1]
        target_matrix = gait_lookup[condition.target_gait][1]
        condition_seed = stable_condition_seed(
            args.eval_seed,
            condition.source_gait,
            condition.source_speed_mps,
            condition.seed_target_speed_mps,
        )
        random.seed(condition_seed)
        np.random.seed(condition_seed)
        torch.manual_seed(condition_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(condition_seed)

        env.reset_idx(all_ids)
        set_fixed_gait(env, condition.source_gait)
        set_fixed_command(env, condition.source_speed_mps)
        cpg_rng = np.random.default_rng(condition_seed)
        cpg_parameters = set_cpg_episode_parameters(env, args, cpg_rng, torch)
        if cpg_parameters["offset_x_control"] != "policy":
            raise SystemExit(
                "Validated OFFSETX_ACTION run unexpectedly reported non-policy "
                f"offset control: {cpg_parameters['offset_x_control']!r}"
            )
        env.compute_observations()
        obs = env.get_observations()

        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        fell_before = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        fell_after = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        fall_step_after = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        action_shape_checked = False

        for _ in range(settle_steps):
            set_fixed_command(env, condition.source_speed_mps)
            with torch.no_grad():
                actions = policy(obs.detach())
                if not action_shape_checked:
                    assert_policy_action_shape(actions, env, action_layout)
                    action_shape_checked = True
                obs, _, _, dones, _ = env.step(actions.detach())
            dones = dones.bool()
            fell_before |= active & dones
            active &= ~dones

        pre_storage: Dict[str, List[np.ndarray]] = {}
        for _ in range(pre_steps):
            set_fixed_command(env, condition.source_speed_mps)
            with torch.no_grad():
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())
            dones = dones.bool()
            valid = active & (~dones)
            fell_before |= active & dones
            active &= ~dones
            gait_eval.append_trace(
                pre_storage,
                **gait_eval.record_state(
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

        # Atomic intervention: update both commands before rebuilding the policy
        # observation.  No physics step or policy action occurs between these
        # assignments, so the first post-switch action sees the target velocity.
        gait_eval.switch_gait_without_reset(env, condition.target_gait)
        set_fixed_command(env, condition.target_speed_mps)
        target_error_at_switch = (
            gait_eval.phase_error(env, target_matrix, torch).detach().cpu().numpy()
        )
        env.compute_observations()
        obs = env.get_observations()

        post_storage: Dict[str, List[np.ndarray]] = {}
        for post_step in range(post_steps):
            set_fixed_command(env, condition.target_speed_mps)
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
            gait_eval.append_trace(
                post_storage,
                **gait_eval.record_state(
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

        pre_trace = gait_eval.stack_trace(pre_storage)
        post_trace = gait_eval.stack_trace(post_storage)
        fell_before_np = fell_before.detach().cpu().numpy()
        fell_after_np = fell_after.detach().cpu().numpy()
        fall_step_np = fall_step_after.detach().cpu().numpy()

        for episode in range(env.num_envs):
            metrics = analyze_episode(
                pre_trace=pre_trace,
                post_trace=post_trace,
                episode=episode,
                condition=condition,
                gait_names=gait_names,
                gait_offsets=gait_offsets,
                args=args,
                dt=dt,
                leg_indices=leg_indices,
                torque_limits=torque_limits,
                spine_index=spine_index if has_active_spine else None,
                robot_mass_kg=robot_mass,
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
                    "action_layout_name": action_layout["action_layout_name"],
                    "offset_action_start": action_layout["offset_action_start"],
                    "offset_action_stop": action_layout["offset_action_stop"],
                    "offset_x_from_policy_actions": 1,
                    "intervention": condition.intervention,
                    "condition_id": condition.condition_id,
                    "comparison_ids": ";".join(condition.comparison_ids),
                    "source_gait": condition.source_gait,
                    "target_gait": condition.target_gait,
                    "source_speed_mps": condition.source_speed_mps,
                    "target_speed_mps": condition.target_speed_mps,
                    "velocity_step_mps": condition.target_speed_mps
                    - condition.source_speed_mps,
                    "gait_command_changed": int(
                        condition.source_gait != condition.target_gait
                    ),
                    "velocity_command_changed": int(
                        not math.isclose(
                            condition.source_speed_mps,
                            condition.target_speed_mps,
                            abs_tol=1.0e-12,
                        )
                    ),
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
            save_timeseries(
                variant_dir / "timeseries" / f"{condition.condition_id}.npz",
                pre_trace,
                post_trace,
                dt,
                condition,
            )

        rows = all_rows[-env.num_envs :]
        mean_iae = np.nanmean(
            [float(row["velocity_error_iae_2s_m"]) for row in rows]
        )
        mean_pitch = np.nanmean(
            [float(row["peak_abs_pitch_change_2s_deg"]) for row in rows]
        )
        print(
            f"[{variant}] {condition.intervention:13s} "
            f"{condition.source_gait}@{condition.source_speed_mps:.1f} -> "
            f"{condition.target_gait}@{condition.target_speed_mps:.1f}: "
            f"IAE_2s={mean_iae:.3f} m, pitch_2s={mean_pitch:.2f} deg"
        )

    write_csv(summary_path, all_rows, SUMMARY_FIELDS)
    metadata = {
        "variant": variant,
        "detected_variant": detected_variant,
        "task": legged.task,
        **control_info,
        **action_layout,
        "validated_action_counts": {
            "rigid": 12,
            "direct_or_phase_locked": 13,
            "uncoupled": 14,
        },
        "requested_command_steps": [
            {
                "comparison_id": step.comparison_id,
                "source_gait": step.source_gait,
                "source_speed_mps": step.source_speed_mps,
                "target_gait": step.target_gait,
                "target_speed_mps": step.target_speed_mps,
            }
            for step in steps
        ],
        "interventions": list(dict.fromkeys(args.interventions)),
        "trial_conditions": [
            {
                "condition_id": condition.condition_id,
                "intervention": condition.intervention,
                "source_gait": condition.source_gait,
                "source_speed_mps": condition.source_speed_mps,
                "target_gait": condition.target_gait,
                "target_speed_mps": condition.target_speed_mps,
                "comparison_ids": list(condition.comparison_ids),
                "condition_seed": stable_condition_seed(
                    args.eval_seed,
                    condition.source_gait,
                    condition.source_speed_mps,
                    condition.seed_target_speed_mps,
                ),
            }
            for condition in conditions
        ],
        "episodes_per_condition": args.episodes,
        "source_settle_s": args.source_settle_s,
        "pre_window_s": args.pre_window_s,
        "post_window_s": args.post_window_s,
        "final_window_s": args.final_window_s,
        "dt_s": dt,
        "eval_seed": args.eval_seed,
        "paired_seed_definition": (
            "source_gait, source_speed_mps, requested_target_speed_mps"
        ),
        "atomic_gait_velocity_update": True,
        "policy_observation_recomputed_before_first_post_switch_action": True,
        "metric_definitions": {
            "normalized_step_response": (
                "(measured_velocity - episode_pre_switch_mean_velocity) / "
                "(target_command - source_command)"
            ),
            "normalized_step_response_2s": (
                "mean normalized response over the final_window_s interval ending at 2 s"
            ),
            "normalized_step_response_final": (
                "mean normalized response over the final_window_s interval at the end "
                "of the post-switch window"
            ),
            "velocity_step_completion_fraction": (
                "normalized step response clipped to [0, 1]"
            ),
            "velocity_overshoot_fraction": (
                "max(0, maximum normalized step response - 1)"
            ),
            "gross_mechanical_work": "integral of sum(abs(applied_torque * joint_velocity))",
            "forward_distance": "integral of max(forward_velocity, 0)",
            "pitch_change": "wrapped pitch relative to the pre-switch circular mean",
        },
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
        "robot_height_range_m": list(map(float, env._cpg.robot_height_range)),
        "ground_clearance_range_m": list(map(float, env._cpg.ground_clearance_range)),
        "ground_penetration_range_m": list(map(float, env._cpg.ground_penetration_range)),
        "offset_x_range_m": list(map(float, env._cpg.offset_x_range)),
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


def to_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_summary_files(input_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(input_dir.glob("**/summary.csv")):
        if path.name == "combined_summary.csv" or "comparison" in path.parts:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(SUMMARY_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise SystemExit(
                    f"{path} is not a gait-velocity summary; missing fields: "
                    f"{sorted(missing)[:8]}"
                )
            rows.extend(reader)
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
    return count, mean, sd, 1.96 * sd / math.sqrt(count)


def aggregate_rows(
    rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        key = (str(row["variant"]), str(row["intervention"]), str(row["condition_id"]))
        groups.setdefault(key, []).append(row)

    base_fields = [
        "variant",
        "intervention",
        "condition_id",
        "comparison_ids",
        "source_gait",
        "target_gait",
        "source_speed_mps",
        "target_speed_mps",
        "velocity_step_mps",
        "episodes",
    ]
    fields = list(base_fields)
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
    for (variant, intervention, condition_id), group in sorted(groups.items()):
        first = group[0]
        record: Dict[str, Any] = {
            "variant": variant,
            "intervention": intervention,
            "condition_id": condition_id,
            "comparison_ids": first["comparison_ids"],
            "source_gait": first["source_gait"],
            "target_gait": first["target_gait"],
            "source_speed_mps": to_float(first["source_speed_mps"]),
            "target_speed_mps": to_float(first["target_speed_mps"]),
            "velocity_step_mps": to_float(first["velocity_step_mps"]),
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


def stable_bootstrap_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def bootstrap_paired_mean(
    differences: np.ndarray,
    resamples: int,
    seed: int,
) -> Tuple[float, float, float, float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(differences))
    median = float(np.median(differences))
    sd = float(np.std(differences, ddof=1)) if differences.size > 1 else math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    boot_means = np.mean(differences[indices], axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean, median, sd, float(low), float(high)


def paired_effect_rows(
    rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant"]), str(row["condition_id"])), []).append(row)

    combined_groups = [
        group
        for group in grouped.values()
        if group and str(group[0]["intervention"]) == "combined"
    ]
    control_groups = [
        group
        for group in grouped.values()
        if group and str(group[0]["intervention"]) == "velocity_only"
    ]
    effects: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    for combined in combined_groups:
        first = combined[0]
        variant = str(first["variant"])
        source_gait = str(first["source_gait"])
        source_speed = to_float(first["source_speed_mps"])
        target_speed = to_float(first["target_speed_mps"])
        candidates = [
            group
            for group in control_groups
            if str(group[0]["variant"]) == variant
            and str(group[0]["source_gait"]) == source_gait
            and math.isclose(
                to_float(group[0]["source_speed_mps"]), source_speed, abs_tol=1.0e-9
            )
            and math.isclose(
                to_float(group[0]["target_speed_mps"]), target_speed, abs_tol=1.0e-9
            )
        ]
        if len(candidates) != 1:
            issues.append(
                {
                    "variant": variant,
                    "combined_condition_id": first["condition_id"],
                    "reason": "missing_velocity_only_control"
                    if not candidates
                    else "ambiguous_velocity_only_control",
                    "detail": f"found {len(candidates)} controls",
                }
            )
            continue
        control = candidates[0]
        combined_by_episode = {to_int(row["episode"]): row for row in combined}
        control_by_episode = {to_int(row["episode"]): row for row in control}
        episode_ids = sorted(set(combined_by_episode) & set(control_by_episode))
        if not episode_ids:
            issues.append(
                {
                    "variant": variant,
                    "combined_condition_id": first["condition_id"],
                    "reason": "no_paired_episodes",
                    "detail": "episode identifiers do not overlap",
                }
            )
            continue
        seed_mismatches = [
            episode
            for episode in episode_ids
            if to_int(combined_by_episode[episode]["condition_seed"])
            != to_int(control_by_episode[episode]["condition_seed"])
        ]
        if seed_mismatches:
            issues.append(
                {
                    "variant": variant,
                    "combined_condition_id": first["condition_id"],
                    "reason": "condition_seed_mismatch",
                    "detail": f"episodes {seed_mismatches[:8]}",
                }
            )
            continue

        for metric in PAIRED_EFFECT_METRICS:
            combined_values: List[float] = []
            control_values: List[float] = []
            paired_ids: List[int] = []
            for episode in episode_ids:
                combined_value = to_float(combined_by_episode[episode].get(metric))
                control_value = to_float(control_by_episode[episode].get(metric))
                if math.isfinite(combined_value) and math.isfinite(control_value):
                    combined_values.append(combined_value)
                    control_values.append(control_value)
                    paired_ids.append(episode)
            if not paired_ids:
                issues.append(
                    {
                        "variant": variant,
                        "combined_condition_id": first["condition_id"],
                        "reason": "no_finite_metric_pairs",
                        "detail": metric,
                    }
                )
                continue
            combined_array = np.asarray(combined_values, dtype=np.float64)
            control_array = np.asarray(control_values, dtype=np.float64)
            differences = combined_array - control_array
            seed = stable_bootstrap_seed(
                args.bootstrap_seed, variant, first["condition_id"], metric
            )
            mean, median, sd, ci_low, ci_high = bootstrap_paired_mean(
                differences, args.bootstrap_resamples, seed
            )
            effects.append(
                {
                    "variant": variant,
                    "comparison_id": first["comparison_ids"],
                    "source_gait": source_gait,
                    "target_gait": first["target_gait"],
                    "source_speed_mps": source_speed,
                    "target_speed_mps": target_speed,
                    "combined_condition_id": first["condition_id"],
                    "control_condition_id": control[0]["condition_id"],
                    "metric": metric,
                    "effect_definition": "combined_minus_velocity_only",
                    "paired_episodes": len(paired_ids),
                    "combined_mean": float(np.mean(combined_array)),
                    "velocity_only_mean": float(np.mean(control_array)),
                    "mean_difference": mean,
                    "median_difference": median,
                    "sd_difference": sd,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ci95_includes_zero": int(ci_low <= 0.0 <= ci_high),
                    "fraction_difference_below_zero": float(np.mean(differences < 0.0)),
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "bootstrap_seed": seed,
                }
            )
    return effects, issues


PAIRED_FIELDS = [
    "variant",
    "comparison_id",
    "source_gait",
    "target_gait",
    "source_speed_mps",
    "target_speed_mps",
    "combined_condition_id",
    "control_condition_id",
    "metric",
    "effect_definition",
    "paired_episodes",
    "combined_mean",
    "velocity_only_mean",
    "mean_difference",
    "median_difference",
    "sd_difference",
    "ci95_low",
    "ci95_high",
    "ci95_includes_zero",
    "fraction_difference_below_zero",
    "bootstrap_resamples",
    "bootstrap_seed",
]


def condition_label(row: Mapping[str, Any]) -> str:
    source = str(row["source_gait"]).replace("_", " ").title()
    target = str(row["target_gait"]).replace("_", " ").title()
    return (
        f"{source} {to_float(row['source_speed_mps']):.1f} -> "
        f"{target} {to_float(row['target_speed_mps']):.1f}"
    )


def make_paired_effect_plots(
    effects: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    if not effects:
        return
    matplotlib_config = output_dir / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV files were written without plots.")
        return

    specifications = (
        (
            "velocity_error_iae_2s_m",
            "Additional velocity IAE over 2 s [m]",
            "additional_velocity_iae_2s.png",
        ),
        (
            "peak_abs_pitch_change_2s_deg",
            "Additional peak pitch excursion over 2 s [deg]",
            "additional_pitch_excursion_2s.png",
        ),
        (
            "requested_leg_torque_saturation_fraction_2s",
            "Additional requested-torque saturation over 2 s",
            "additional_torque_saturation_2s.png",
        ),
        (
            "total_gross_work_2s_j",
            "Additional gross mechanical work over 2 s [J]",
            "additional_gross_work_2s.png",
        ),
        (
            "work_per_forward_m_2s_jpm",
            "Additional work per forward distance over 2 s [J/m]",
            "additional_work_per_distance_2s.png",
        ),
    )
    preferred_order = ("direct_angle", "rigid", "uncoupled")
    variants_present = {str(row["variant"]) for row in effects}
    variants = [name for name in preferred_order if name in variants_present]
    variants.extend(sorted(variants_present - set(variants)))
    markers = ("o", "s", "^")
    colors = ("#0072B2", "#D55E00", "#009E73")

    for metric, ylabel, filename in specifications:
        selected = [row for row in effects if str(row["metric"]) == metric]
        if not selected:
            continue
        condition_ids = list(
            dict.fromkeys(str(row["comparison_id"]) for row in selected)
        )
        labels = []
        for condition_id in condition_ids:
            labels.append(
                condition_label(
                    next(
                        row
                        for row in selected
                        if row["comparison_id"] == condition_id
                    )
                )
            )
        x = np.arange(len(condition_ids), dtype=np.float64)
        fig, axis = plt.subplots(
            figsize=(max(9.0, 1.55 * len(condition_ids)), 4.8)
        )
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.75)
        for variant_index, variant in enumerate(variants):
            offset = (variant_index - 0.5 * (len(variants) - 1)) * 0.16
            points = {
                str(row["comparison_id"]): row
                for row in selected
                if str(row["variant"]) == variant
            }
            y = np.asarray(
                [
                    to_float(points.get(condition_id, {}).get("mean_difference"))
                    for condition_id in condition_ids
                ]
            )
            low = np.asarray(
                [
                    to_float(points.get(condition_id, {}).get("ci95_low"))
                    for condition_id in condition_ids
                ]
            )
            high = np.asarray(
                [
                    to_float(points.get(condition_id, {}).get("ci95_high"))
                    for condition_id in condition_ids
                ]
            )
            finite = np.isfinite(y)
            if not np.any(finite):
                continue
            axis.errorbar(
                (x + offset)[finite],
                y[finite],
                yerr=np.vstack(
                    (
                        np.maximum(0.0, y[finite] - low[finite]),
                        np.maximum(0.0, high[finite] - y[finite]),
                    )
                ),
                fmt=markers[variant_index % len(markers)],
                color=colors[variant_index % len(colors)],
                capsize=3,
                linewidth=1.3,
                label=variant,
            )
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Combined command step")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def ensure_compare_output_writable(output_dir: Path, overwrite: bool) -> None:
    protected = (
        output_dir / "combined_summary.csv",
        output_dir / "aggregate_by_variant_condition.csv",
        output_dir / "paired_combined_minus_velocity_only.csv",
    )
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"Comparison output already exists ({existing[0]}); pass --overwrite to replace it"
        )


def run_comparison(args: argparse.Namespace) -> None:
    validate_args(args)
    input_dir = args.input_dir or args.output_dir
    output_dir = args.output_dir
    rows = read_summary_files(input_dir)
    if not rows:
        raise SystemExit(f"No gait-velocity summary.csv files found under {input_dir}")
    ensure_compare_output_writable(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "combined_summary.csv", rows, SUMMARY_FIELDS)
    aggregate, aggregate_fields = aggregate_rows(rows)
    write_csv(
        output_dir / "aggregate_by_variant_condition.csv",
        aggregate,
        aggregate_fields,
    )
    effects, issues = paired_effect_rows(rows, args)
    write_csv(
        output_dir / "paired_combined_minus_velocity_only.csv",
        effects,
        PAIRED_FIELDS,
    )
    issue_fields = ("variant", "combined_condition_id", "reason", "detail")
    write_csv(output_dir / "comparison_issues.csv", issues, issue_fields)
    make_paired_effect_plots(effects, output_dir)

    analysis_metadata = {
        "effect_definition": "combined gait+velocity minus matched velocity-only",
        "pairing_fields": [
            "variant",
            "source_gait",
            "source_speed_mps",
            "target_speed_mps",
            "episode",
        ],
        "condition_seed_equality_required": True,
        "bootstrap": {
            "type": "paired episode-level percentile bootstrap of the mean difference",
            "resamples": args.bootstrap_resamples,
            "base_seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "variants": sorted({str(row["variant"]) for row in rows}),
        "episode_rows": len(rows),
        "aggregate_conditions": len(aggregate),
        "paired_effect_rows": len(effects),
        "issues": len(issues),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Combined {len(rows)} episode rows from "
        f"{len(set(row['variant'] for row in rows))} variants; "
        f"wrote {len(effects)} paired effects and {len(issues)} quality issues."
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
