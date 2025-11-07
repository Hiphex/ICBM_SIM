"""
Simple ICBM launch and interception simulation.

This module models a ballistic missile trajectory with gravity, atmospheric drag,
an optional boost phase, and a guided interceptor that launches after a configurable
delay. The guidance law steers towards the noisy line of sight and is limited by a
maximum lateral acceleration. The equations of motion are integrated with a fixed time
step; the default parameters yield an intercept, but Monte Carlo mode exposes failure
cases when sensor noise, wind, or slower interceptor kinematics are sampled.

Run the module directly to execute a default single run:

    python simulation.py

Use ``--plot`` to show the 2D trajectories (requires matplotlib) or ``--runs N`` to
perform N Monte Carlo samples and view aggregate statistics.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Optional, Tuple
import json
import random
import statistics
from pathlib import Path


Vector = Tuple[float, float]

DEFAULT_BOOST_PROFILE: Tuple[Tuple[float, float], ...] = (
    (55.0, 16.0),
    (65.0, 9.0),
    (80.0, 3.5),
)

DEFAULT_PITCH_SCHEDULE_DEG: Tuple[float, ...] = (0.0, -5.0, -12.0)


def add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1]


def mul(a: Vector, scalar: float) -> Vector:
    return a[0] * scalar, a[1] * scalar


def length(a: Vector) -> float:
    return math.hypot(a[0], a[1])


def normalize(a: Vector) -> Vector:
    mag = length(a)
    if mag == 0:
        return 0.0, 0.0
    return a[0] / mag, a[1] / mag


def rotate(a: Vector, angle_rad: float) -> Vector:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (a[0] * cos_a - a[1] * sin_a, a[0] * sin_a + a[1] * cos_a)


@dataclass
class TrajectorySample:
    time: float
    icbm_position: Vector
    icbm_velocity: Vector
    interceptor_position: Optional[Vector]
    interceptor_velocity: Optional[Vector]
    decoy_positions: List[Vector]
    decoy_ids: List[int]
    interceptor_positions_map: Dict[str, Optional[Vector]] = field(default_factory=dict)
    interceptor_velocities_map: Dict[str, Optional[Vector]] = field(default_factory=dict)


@dataclass
class SimulationResult:
    intercept_success: bool
    intercept_time: Optional[float]
    intercept_position: Optional[Vector]
    icbm_impact_time: Optional[float]
    samples: List[TrajectorySample]
    intercept_target_label: Optional[str]
    decoy_count: int
    parameters: Dict[str, Any]
    interceptor_reports: Dict[str, "InterceptorReport"]


@dataclass(frozen=True)
class InterceptorConfig:
    name: str
    site: Vector
    launch_delay: float
    engage_altitude_min: float
    engage_altitude_max: float
    speed_cap: float
    guidance_gain: float
    damping_gain: float
    intercept_distance: float
    max_accel: float
    guidance_noise_std_deg: float
    confusion_probability: float
    reacquisition_rate: float
    max_flight_time: float
    depends_on: Optional[str] = None
    dependency_grace_period: float = 0.0
    salvo_count: int = 1
    salvo_interval: float = 0.0
    engage_range_min: float = 0.0
    engage_range_max: float = 0.0


@dataclass
class InterceptorState:
    config: InterceptorConfig
    salvo_index: int
    label: str
    planned_launch_time: float
    launched: bool = False
    expended: bool = False
    position: Optional[Vector] = None
    velocity: Optional[Vector] = None
    launch_time: Optional[float] = None
    intercept_time: Optional[float] = None
    intercept_position: Optional[Vector] = None
    intercept_target_label: Optional[str] = None
    success: bool = False
    target_mode: str = "primary"
    selected_decoy_index: Optional[int] = None


@dataclass
class InterceptorReport:
    name: str
    config_name: str
    salvo_index: int
    success: bool
    target_label: Optional[str]
    intercept_time: Optional[float]
    intercept_position: Optional[Vector]
    launch_time: Optional[float]
    expended: bool


@dataclass
class MonteCarloSummary:
    runs: int
    successes: int
    impacts: int
    timeouts: int
    intercept_times: List[float]
    miss_distances: List[float]
    decoy_intercepts: int
    layer_primary_kills: Dict[str, int] = field(default_factory=dict)
    layer_decoy_hits: Dict[str, int] = field(default_factory=dict)

    def to_report(self) -> str:
        lines = [
            f"Monte Carlo runs: {self.runs}",
            f"  Successes: {self.successes}",
            f"  Impacts:   {self.impacts}",
            f"  Timeouts:  {self.timeouts}",
        ]
        if self.decoy_intercepts:
            lines.append(f"  Intercepts on decoy targets: {self.decoy_intercepts}")
        if self.layer_primary_kills:
            layer_summary = ", ".join(
                f"{name}: {count}" for name, count in sorted(self.layer_primary_kills.items())
            )
            lines.append(f"  Primary kills by layer: {layer_summary}")
        if self.layer_decoy_hits:
            decoy_summary = ", ".join(
                f"{name}: {count}" for name, count in sorted(self.layer_decoy_hits.items())
            )
            lines.append(f"  Decoy hits by layer: {decoy_summary}")
        if self.intercept_times:
            avg = statistics.mean(self.intercept_times)
            stdev = statistics.pstdev(self.intercept_times) if len(self.intercept_times) > 1 else 0.0
            lines.append(f"  Avg intercept time: {avg:6.1f}s (std {stdev:5.1f}s)")
        if self.miss_distances:
            median_miss = statistics.median(self.miss_distances)
            lines.append(f"  Median terminal miss distance: {median_miss:,.0f} m")
        return "\n".join(lines)


def _interceptor_style(config_name: str) -> Dict[str, Any]:
    """Return consistent style hints for a given interceptor family."""
    base_styles: Dict[str, Dict[str, Any]] = {
        "GBI": {"color": "#1E88E5", "linewidth": 2.3},
        "THAAD": {"color": "#D81B60", "linewidth": 2.3, "linestyle": "-."},
    }
    return dict(base_styles.get(config_name, {}))


def simulate_icbm_intercept(
    *,
    dt: float = 0.25,
    max_time: Optional[float] = None,
    gravity: float = 9.81,
    icbm_start: Vector = (0.0, 0.0),
    icbm_velocity: Vector = (2400.0, 6200.0),
    interceptor_site: Vector = (3_800_000.0, 0.0),
    interceptor_speed_cap: float = 5400.0,
    interceptor_launch_delay: float = 120.0,
    gbi_salvo_count: int = 1,
    gbi_salvo_interval: float = 0.0,
    thaad_salvo_count: int = 1,
    thaad_salvo_interval: float = 0.0,
    guidance_gain: float = 0.7,
    damping_gain: float = 0.05,
    intercept_distance: float = 300.0,
    icbm_mass: float = 30_000.0,
    icbm_drag_coefficient: float = 0.12,
    icbm_reference_area: float = 3.5,
    atmospheric_density_sea_level: float = 1.225,
    atmospheric_scale_height: float = 8500.0,
    icbm_boost_profile: Tuple[Tuple[float, float], ...] = DEFAULT_BOOST_PROFILE,
    icbm_pitch_schedule_deg: Tuple[float, ...] = DEFAULT_PITCH_SCHEDULE_DEG,
    wind_velocity: Vector = (0.0, 0.0),
    guidance_noise_std_deg: float = 0.10,
    interceptor_max_accel: float = 60.0,
    decoy_release_time: Optional[float] = 220.0,
    decoy_count: int = 3,
    decoy_spread_velocity: float = 280.0,
    decoy_drag_multiplier: float = 4.0,
    warhead_mass_fraction: float = 0.35,
    warhead_drag_multiplier: float = 0.6,
    decoy_mass_fraction: float = 0.04,
    decoy_confusion_probability: float = 0.1,
    decoy_reacquisition_rate: float = 0.015,
    rng: Optional[Random] = None,
    interceptors: Optional[List[InterceptorConfig]] = None,
) -> SimulationResult:
    """
    Simulate a ballistic launch and interception attempt with optional drag,
    throttle-limited boost and noisy guidance measurements.

    The model still uses a 2D plane (range/altitude) but introduces a set of
    knobs that make experimentation easier:
    * Multi-stage boost sequence: each stage carries a thrust duration and lateral
      acceleration, applied along a scheduled pitch angle, before the vehicle
      transitions into mid-course ballistic flight with atmosphere-driven drag.
    * Constant wind that influences the missile's relative airspeed.
    * Interceptor guidance that can be limited by maximum acceleration and
      perturbed by line-of-sight measurement noise, which gives room for
      Monte-Carlo style studies.
    * Optional mid-course decoy deployment with configurable seeker confusion
      probability and reacquisition rate, allowing Monte Carlo runs to capture
      false kills on decoys versus the primary warhead.
    * Configurable interceptor salvos (count and spacing) so layered defenses can
      fire parallel shots for higher kill probability.
    """
    if rng is None:
        rng = random.Random()

    initial_heading = normalize(icbm_velocity)
    if initial_heading == (0.0, 0.0):
        initial_heading = (0.0, 1.0)

    boost_profile: Tuple[Tuple[float, float], ...] = tuple(
        (max(0.0, duration), accel) for duration, accel in icbm_boost_profile if duration > 0.0
    )
    cumulative_stage_times: List[float] = []
    stage_accels: List[float] = []
    total_duration = 0.0
    for duration, accel in boost_profile:
        total_duration += duration
        cumulative_stage_times.append(total_duration)
        stage_accels.append(accel)

    stage_count = len(boost_profile)
    pitch_schedule = list(icbm_pitch_schedule_deg[:stage_count])
    if stage_count and not pitch_schedule:
        pitch_schedule = [0.0 for _ in range(stage_count)]
    if len(pitch_schedule) < stage_count:
        last_angle = pitch_schedule[-1] if pitch_schedule else 0.0
        pitch_schedule.extend([last_angle] * (stage_count - len(pitch_schedule)))

    cumulative_pitch_angles: List[float] = []
    running_angle = 0.0
    for idx in range(stage_count):
        running_angle += pitch_schedule[idx]
        cumulative_pitch_angles.append(running_angle)

    normalized_gbi_salvo_count = max(1, int(gbi_salvo_count))
    normalized_gbi_salvo_interval = max(0.0, gbi_salvo_interval)
    normalized_thaad_salvo_count = max(1, int(thaad_salvo_count))
    normalized_thaad_salvo_interval = max(0.0, thaad_salvo_interval)

    if interceptors is None:
        interceptors = [
            InterceptorConfig(
                name="GBI",
                site=interceptor_site,
                launch_delay=interceptor_launch_delay,
                engage_altitude_min=120_000.0,
                engage_altitude_max=1_200_000.0,
                engage_range_min=400_000.0,
                engage_range_max=6_000_000.0,
                speed_cap=interceptor_speed_cap,
                guidance_gain=max(0.88, guidance_gain * 1.18),
                damping_gain=max(0.055, damping_gain * 1.15),
                intercept_distance=max(intercept_distance, 50_000.0),
                max_accel=interceptor_max_accel,
                guidance_noise_std_deg=max(0.03, guidance_noise_std_deg * 0.9),
                confusion_probability=max(0.0, min(0.15, decoy_confusion_probability * 0.5)),
                reacquisition_rate=max(decoy_reacquisition_rate, 0.018),
                max_flight_time=900.0,
                depends_on=None,
                dependency_grace_period=0.0,
                salvo_count=normalized_gbi_salvo_count,
                salvo_interval=normalized_gbi_salvo_interval,
            ),
            InterceptorConfig(
                name="THAAD",
                site=interceptor_site,
                launch_delay=interceptor_launch_delay + 220.0,
                engage_altitude_min=20_000.0,
                engage_altitude_max=220_000.0,
                engage_range_min=0.0,
                engage_range_max=260_000.0,
                speed_cap=5000.0,
                guidance_gain=max(0.68, guidance_gain * 1.38),
                damping_gain=max(0.11, damping_gain * 1.9),
                intercept_distance=180_000.0,
                max_accel=max(interceptor_max_accel, 155.0),
                guidance_noise_std_deg=max(0.035, guidance_noise_std_deg * 1.05),
                confusion_probability=min(0.12, decoy_confusion_probability + 0.03),
                reacquisition_rate=max(decoy_reacquisition_rate * 2.0, 0.06),
                max_flight_time=560.0,
                depends_on="GBI",
                dependency_grace_period=45.0,
                salvo_count=normalized_thaad_salvo_count,
                salvo_interval=normalized_thaad_salvo_interval,
            ),
        ]

    interceptor_states: List[InterceptorState] = []
    for cfg in interceptors:
        salvo_count = max(1, int(cfg.salvo_count))
        interval = max(0.0, cfg.salvo_interval)
        for salvo_index in range(salvo_count):
            label = cfg.name if salvo_count == 1 else f"{cfg.name}#{salvo_index + 1}"
            planned_launch_time = cfg.launch_delay + salvo_index * interval
            interceptor_states.append(
                InterceptorState(
                    config=cfg,
                    salvo_index=salvo_index,
                    label=label,
                    planned_launch_time=planned_launch_time,
                )
            )

    parameter_record: Dict[str, Any] = {
        "dt": dt,
        "max_time": max_time,
        "gravity": gravity,
        "icbm_start": icbm_start,
        "icbm_initial_velocity": icbm_velocity,
        "interceptor_site": interceptor_site,
        "interceptor_speed_cap": interceptor_speed_cap,
        "interceptor_launch_delay": interceptor_launch_delay,
        "gbi_salvo_count": normalized_gbi_salvo_count,
        "gbi_salvo_interval": normalized_gbi_salvo_interval,
        "thaad_salvo_count": normalized_thaad_salvo_count,
        "thaad_salvo_interval": normalized_thaad_salvo_interval,
        "guidance_gain": guidance_gain,
        "damping_gain": damping_gain,
        "intercept_distance": intercept_distance,
        "icbm_mass": icbm_mass,
        "icbm_drag_coefficient": icbm_drag_coefficient,
        "icbm_reference_area": icbm_reference_area,
        "atmospheric_density_sea_level": atmospheric_density_sea_level,
        "atmospheric_scale_height": atmospheric_scale_height,
        "icbm_boost_profile": boost_profile,
        "icbm_pitch_schedule_deg": tuple(pitch_schedule),
        "wind_velocity": wind_velocity,
        "guidance_noise_std_deg": guidance_noise_std_deg,
        "interceptor_max_accel": interceptor_max_accel,
        "decoy_release_time": decoy_release_time,
        "decoy_count": decoy_count,
        "decoy_spread_velocity": decoy_spread_velocity,
        "decoy_drag_multiplier": decoy_drag_multiplier,
        "warhead_mass_fraction": warhead_mass_fraction,
        "warhead_drag_multiplier": warhead_drag_multiplier,
        "decoy_mass_fraction": decoy_mass_fraction,
        "decoy_confusion_probability": decoy_confusion_probability,
        "decoy_reacquisition_rate": decoy_reacquisition_rate,
        "interceptors": [
            {
                "name": cfg.name,
                "site": cfg.site,
                "launch_delay": cfg.launch_delay,
                "engage_altitude_min": cfg.engage_altitude_min,
                "engage_altitude_max": cfg.engage_altitude_max,
                "engage_range_min": cfg.engage_range_min,
                "engage_range_max": cfg.engage_range_max,
                "speed_cap": cfg.speed_cap,
                "guidance_gain": cfg.guidance_gain,
                "damping_gain": cfg.damping_gain,
                "intercept_distance": cfg.intercept_distance,
                "max_accel": cfg.max_accel,
                "guidance_noise_std_deg": cfg.guidance_noise_std_deg,
                "confusion_probability": cfg.confusion_probability,
                "reacquisition_rate": cfg.reacquisition_rate,
                "max_flight_time": cfg.max_flight_time,
                "depends_on": cfg.depends_on,
                "dependency_grace_period": cfg.dependency_grace_period,
                "salvo_count": cfg.salvo_count,
                "salvo_interval": cfg.salvo_interval,
            }
            for cfg in interceptors
        ],
    }

    def air_density(altitude: float) -> float:
        if altitude <= 0.0:
            return atmospheric_density_sea_level
        return atmospheric_density_sea_level * math.exp(-altitude / atmospheric_scale_height)

    # Internal state copies so we do not mutate caller defaults.
    icbm_pos = icbm_start
    icbm_vel = icbm_velocity

    samples: List[TrajectorySample] = []

    intercept_success = False
    intercept_time: Optional[float] = None
    intercept_position: Optional[Vector] = None
    icbm_impact_time: Optional[float] = None
    intercept_target_label: Optional[str] = None

    active_drag_coefficient = icbm_drag_coefficient
    active_reference_area = icbm_reference_area
    active_mass = icbm_mass

    decoy_positions: List[Vector] = []
    decoy_velocities: List[Vector] = []
    decoy_masses: List[float] = []
    decoy_drag_coefficients: List[float] = []
    decoy_reference_areas: List[float] = []
    decoy_ids: List[int] = []
    next_decoy_id = 0
    decoys_deployed = False
    released_decoy_count = 0

    def snapshot_decoys() -> List[Vector]:
        return [
            (pos[0], pos[1] if pos[1] >= 0.0 else 0.0)
            for pos in decoy_positions
        ]

    time = 0.0
    step_count = 0
    max_steps = math.inf
    if max_time is not None:
        max_steps = int(math.ceil(max_time / dt))
    else:
        # Safety guard: with dt=0.25, 200k steps ≈ 50,000 seconds (~13.9 hours).
        max_steps = 200_000

    while True:
        if (
            not decoys_deployed
            and decoy_release_time is not None
            and decoy_count > 0
            and time >= decoy_release_time
        ):
            decoys_deployed = True

            active_mass = max(1.0, icbm_mass * max(warhead_mass_fraction, 0.05))
            active_drag_coefficient = icbm_drag_coefficient * warhead_drag_multiplier
            active_reference_area = icbm_reference_area * warhead_drag_multiplier

            decoy_positions = []
            decoy_velocities = []
            decoy_masses = []
            decoy_drag_coefficients = []
            decoy_reference_areas = []
            decoy_ids = []

            for _ in range(decoy_count):
                orientation = normalize((rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)))
                if orientation == (0.0, 0.0):
                    orientation = (1.0, 0.0)
                speed_offset = max(
                    0.0, rng.gauss(decoy_spread_velocity, max(20.0, 0.25 * decoy_spread_velocity))
                )
                lateral_velocity = mul(orientation, speed_offset)
                vertical_offset = (0.0, -abs(rng.gauss(0.0, 0.2 * max(decoy_spread_velocity, 1.0))))
                new_velocity = add(icbm_vel, add(lateral_velocity, vertical_offset))

                decoy_positions.append(icbm_pos)
                decoy_velocities.append(new_velocity)
                decoy_masses.append(max(1.0, icbm_mass * max(decoy_mass_fraction, 0.01)))
                drag_multiplier = max(decoy_drag_multiplier, 0.1)
                decoy_drag_coefficients.append(icbm_drag_coefficient * drag_multiplier)
                decoy_reference_areas.append(icbm_reference_area * drag_multiplier)
                decoy_ids.append(next_decoy_id)
                next_decoy_id += 1

            released_decoy_count = len(decoy_positions)

            if decoy_positions:
                for state in interceptor_states:
                    if state.expended:
                        continue
                    if state.launched and state.position is not None:
                        confused = rng.random() < state.config.confusion_probability
                    else:
                        confused = False
                    if confused:
                        state.target_mode = "decoy"
                        state.selected_decoy_index = rng.randrange(len(decoy_positions))
                    else:
                        state.target_mode = "primary"
                        state.selected_decoy_index = None

        gravity_vec = (0.0, -gravity)
        rel_air_velocity = sub(icbm_vel, wind_velocity)
        rel_speed = length(rel_air_velocity)

        drag_vec = (0.0, 0.0)
        if rel_speed > 0.0 and active_mass > 0.0:
            rho = air_density(icbm_pos[1])
            drag_mag = (
                0.5
                * rho
                * rel_speed
                * rel_speed
                * active_drag_coefficient
                * active_reference_area
                / active_mass
            )
            drag_vec = mul(normalize(rel_air_velocity), -drag_mag)

        boost_vec = (0.0, 0.0)
        if stage_count:
            stage_idx = None
            for idx, stage_end in enumerate(cumulative_stage_times):
                if time < stage_end:
                    stage_idx = idx
                    break
            if stage_idx is not None:
                angle_deg = cumulative_pitch_angles[stage_idx]
                heading_vec = rotate(initial_heading, math.radians(angle_deg))
                heading_vec = normalize(heading_vec)
                boost_vec = mul(heading_vec, stage_accels[stage_idx])

        total_acc = add(add(gravity_vec, drag_vec), boost_vec)
        icbm_vel = add(icbm_vel, mul(total_acc, dt))
        icbm_pos = add(icbm_pos, mul(icbm_vel, dt))

        for idx in range(len(decoy_positions)):
            decoy_velocity = decoy_velocities[idx]
            rel_air = sub(decoy_velocity, wind_velocity)
            rel_speed_decoy = length(rel_air)
            decoy_drag_vec = (0.0, 0.0)
            decoy_mass = decoy_masses[idx]
            if rel_speed_decoy > 0.0 and decoy_mass > 0.0:
                rho = air_density(decoy_positions[idx][1])
                drag_mag_decoy = (
                    0.5
                    * rho
                    * rel_speed_decoy
                    * rel_speed_decoy
                    * decoy_drag_coefficients[idx]
                    * decoy_reference_areas[idx]
                    / decoy_mass
                )
                decoy_drag_vec = mul(normalize(rel_air), -drag_mag_decoy)

            total_decoy_acc = add((0.0, -gravity), decoy_drag_vec)
            decoy_velocity = add(decoy_velocity, mul(total_decoy_acc, dt))
            decoy_position = add(decoy_positions[idx], mul(decoy_velocity, dt))

            decoy_velocities[idx] = decoy_velocity
            decoy_positions[idx] = decoy_position

        target_altitude = icbm_pos[1]
        for state in interceptor_states:
            if state.launched or state.expended:
                continue
            cfg = state.config
            if intercept_success:
                continue
            horiz_distance = abs(icbm_pos[0] - cfg.site[0])
            range_ok = True
            if cfg.engage_range_min > 0.0 and horiz_distance < cfg.engage_range_min:
                range_ok = False
            if cfg.engage_range_max > 0.0 and horiz_distance > cfg.engage_range_max:
                range_ok = False
            dependency_ready = True
            if cfg.depends_on:
                # Delay launch until every interceptor in the dependency layer has failed
                # or exceeded its grace window.
                dependency_states = [
                    s for s in interceptor_states if s.config.name == cfg.depends_on
                ]
                if dependency_states:
                    if any(dep.success for dep in dependency_states):
                        continue
                    dependency_ready = True
                    grace = max(cfg.dependency_grace_period, 0.0)
                    for dep in dependency_states:
                        if dep.expended and not dep.success:
                            continue
                        if dep.launched:
                            if grace > 0.0 and dep.launch_time is not None:
                                if time - dep.launch_time >= grace:
                                    continue
                            dependency_ready = False
                            break
                        else:
                            if grace > 0.0 and time >= dep.planned_launch_time + grace:
                                continue
                            dependency_ready = False
                            break
                else:
                    dependency_ready = True
            if not dependency_ready:
                continue
            if (
                time >= state.planned_launch_time
                and cfg.engage_altitude_min <= target_altitude <= cfg.engage_altitude_max
                and range_ok
            ):
                state.launched = True
                state.position = cfg.site
                state.velocity = (0.0, 0.0)
                state.launch_time = time
                if decoys_deployed and decoy_positions and rng.random() < cfg.confusion_probability:
                    state.target_mode = "decoy"
                    state.selected_decoy_index = rng.randrange(len(decoy_positions))
                else:
                    state.target_mode = "primary"
                    state.selected_decoy_index = None

        for state in interceptor_states:
            if (
                not state.launched
                or state.expended
                or state.position is None
                or state.velocity is None
            ):
                continue

            cfg = state.config

            if cfg.max_flight_time > 0.0 and state.launch_time is not None:
                if time - state.launch_time > cfg.max_flight_time:
                    state.expended = True
                    continue

            if decoys_deployed and state.target_mode == "decoy" and decoy_positions:
                reacquire_prob = (
                    1.0 - math.exp(-cfg.reacquisition_rate * dt)
                    if cfg.reacquisition_rate > 0.0
                    else 0.0
                )
                if rng.random() < reacquire_prob:
                    state.target_mode = "primary"
                    state.selected_decoy_index = None

            target_pos = icbm_pos
            if state.target_mode == "decoy" and decoy_positions:
                idx = state.selected_decoy_index
                if idx is None or idx < 0 or idx >= len(decoy_positions):
                    idx = len(decoy_positions) - 1
                if idx >= 0:
                    target_pos = decoy_positions[idx]
                    state.selected_decoy_index = idx
                else:
                    state.target_mode = "primary"
                    state.selected_decoy_index = None
                    target_pos = icbm_pos

            line_of_sight = sub(target_pos, state.position)
            los_direction = normalize(line_of_sight)

            if cfg.guidance_noise_std_deg > 0.0:
                noise_angle = math.radians(rng.gauss(0.0, cfg.guidance_noise_std_deg))
                los_direction = rotate(los_direction, noise_angle)
                los_direction = normalize(los_direction)

            desired_velocity = mul(los_direction, cfg.speed_cap)
            guidance_acc = mul(sub(desired_velocity, state.velocity), cfg.guidance_gain)
            damping = mul(state.velocity, cfg.damping_gain)
            control_acc = sub(guidance_acc, damping)

            acc_mag = length(control_acc)
            if acc_mag > cfg.max_accel:
                control_acc = mul(control_acc, cfg.max_accel / acc_mag)

            state.velocity = add(state.velocity, mul(control_acc, dt))

            speed = length(state.velocity)
            if speed > cfg.speed_cap:
                state.velocity = mul(normalize(state.velocity), cfg.speed_cap)

            state.position = add(state.position, mul(state.velocity, dt))

            primary_distance = length(sub(icbm_pos, state.position))
            decoy_hit_index: Optional[int] = None
            if decoy_positions and state.target_mode == "decoy":
                for idx, decoy_position in enumerate(decoy_positions):
                    if length(sub(decoy_position, state.position)) <= cfg.intercept_distance:
                        decoy_hit_index = idx
                        break

            if primary_distance <= cfg.intercept_distance:
                intercept_success = True
                intercept_time = time
                intercept_position = icbm_pos
                intercept_target_label = "primary"
                state.success = True
                state.expended = True
                state.intercept_time = time
                state.intercept_position = icbm_pos
                state.intercept_target_label = "primary"
                state.position = icbm_pos
                state.velocity = (0.0, 0.0)
                break
            if decoy_hit_index is not None:
                intercept_time = time
                target_pos = decoy_positions[decoy_hit_index] if decoy_hit_index < len(decoy_positions) else state.position
                intercept_position = target_pos
                intercept_target_label = "decoy"
                state.success = False
                state.expended = True
                state.intercept_time = time
                state.intercept_position = target_pos
                state.intercept_target_label = "decoy"
                state.position = target_pos
                state.velocity = (0.0, 0.0)
                if decoy_hit_index < len(decoy_positions):
                    decoy_positions.pop(decoy_hit_index)
                    decoy_velocities.pop(decoy_hit_index)
                    decoy_masses.pop(decoy_hit_index)
                    decoy_drag_coefficients.pop(decoy_hit_index)
                    decoy_reference_areas.pop(decoy_hit_index)
                    decoy_ids.pop(decoy_hit_index)
                for other_state in interceptor_states:
                    other_state.selected_decoy_index = None
                break

        if icbm_pos[1] < 0.0:
            icbm_pos = (icbm_pos[0], 0.0)

        for idx, pos in enumerate(decoy_positions):
            if pos[1] < 0.0:
                decoy_positions[idx] = (pos[0], 0.0)
                decoy_velocities[idx] = (decoy_velocities[idx][0], 0.0)

        interceptor_positions_map = {
            state.label: state.position for state in interceptor_states
        }
        interceptor_velocities_map = {
            state.label: state.velocity for state in interceptor_states
        }

        default_interceptor_position = None
        default_interceptor_velocity = None
        if interceptors:
            default_label = next(
                (state.label for state in interceptor_states if state.config is interceptors[0]),
                None,
            )
            if default_label is not None:
                default_interceptor_position = interceptor_positions_map.get(default_label)
                default_interceptor_velocity = interceptor_velocities_map.get(default_label)

        decoy_snapshot = snapshot_decoys()

        samples.append(
            TrajectorySample(
                time,
                icbm_position=icbm_pos,
                icbm_velocity=icbm_vel,
                interceptor_position=default_interceptor_position,
                interceptor_velocity=default_interceptor_velocity,
                interceptor_positions_map=interceptor_positions_map,
                interceptor_velocities_map=interceptor_velocities_map,
                decoy_positions=decoy_snapshot,
                decoy_ids=list(decoy_ids),
            )
        )

        if intercept_success:
            break

        if icbm_pos[1] <= 0.0 and time > 0.0:
            icbm_impact_time = time
            break

        time += dt
        step_count += 1
        if step_count >= max_steps:
            if max_time is not None:
                break
            raise RuntimeError(
                "Simulation exceeded safety iteration limit without intercept or ground impact. "
                "Check parameters for runaway conditions or supply max_time."
            )

    parameter_record["decoys_deployed"] = released_decoy_count
    parameter_record["warhead_active_mass"] = active_mass
    parameter_record["warhead_active_drag_coefficient"] = active_drag_coefficient
    parameter_record["warhead_active_reference_area"] = active_reference_area

    interceptor_reports = {
        state.label: InterceptorReport(
            name=state.label,
            config_name=state.config.name,
            salvo_index=state.salvo_index,
            success=state.success,
            target_label=state.intercept_target_label,
            intercept_time=state.intercept_time,
            intercept_position=state.intercept_position,
            launch_time=state.launch_time,
            expended=state.expended,
        )
        for state in interceptor_states
    }

    return SimulationResult(
        intercept_success=intercept_success,
        intercept_time=intercept_time,
        intercept_position=intercept_position,
        icbm_impact_time=icbm_impact_time,
        samples=samples,
        intercept_target_label=intercept_target_label,
        decoy_count=released_decoy_count,
        parameters=parameter_record,
        interceptor_reports=interceptor_reports,
    )


def _minimum_miss_distance(result: SimulationResult) -> float:
    distances: List[float] = []
    for sample in result.samples:
        if sample.interceptor_positions_map:
            for pos in sample.interceptor_positions_map.values():
                if pos is None:
                    continue
                distances.append(length(sub(sample.icbm_position, pos)))
        elif sample.interceptor_position is not None:
            distances.append(length(sub(sample.icbm_position, sample.interceptor_position)))
    if distances:
        return min(distances)
    if result.samples:
        # Fall back to the final range to origin if interceptor never launched.
        terminal = result.samples[-1]
        return length(terminal.icbm_position)
    return math.inf


def _result_to_entry(
    result: SimulationResult,
    *,
    mode: str,
    run_index: int,
    seed: Optional[int],
    min_distance: Optional[float] = None,
) -> Dict[str, Any]:
    if min_distance is None:
        min_distance = _minimum_miss_distance(result)
    entry: Dict[str, Any] = {
        "mode": mode,
        "run": run_index,
        "seed": seed,
        "success": result.intercept_success,
        "target": result.intercept_target_label,
        "intercept_time": result.intercept_time,
        "impact_time": result.icbm_impact_time,
        "decoy_count": result.decoy_count,
        "intercept_position": result.intercept_position,
        "min_primary_distance": min_distance,
        "parameters": dict(result.parameters),
    }
    if entry["intercept_position"] is not None:
        entry["intercept_position"] = list(entry["intercept_position"])
    entry["interceptors"] = {
        name: {
            "config_name": report.config_name,
            "salvo_index": report.salvo_index,
            "success": report.success,
            "target": report.target_label,
            "intercept_time": report.intercept_time,
            "intercept_position": list(report.intercept_position) if report.intercept_position else None,
            "launch_time": report.launch_time,
            "expended": report.expended,
        }
        for name, report in result.interceptor_reports.items()
    }
    return entry


def _summarize(result: SimulationResult) -> str:
    if result.intercept_success and result.intercept_time is not None:
        x, y = result.intercept_position or (0.0, 0.0)
        interceptor_name = None
        for name, report in result.interceptor_reports.items():
            if report.success and report.target_label == "primary":
                interceptor_name = name
                break
        prefix = f"{interceptor_name} interceptor" if interceptor_name else "Interceptor"
        return (
            f"{prefix} achieved lock at t={result.intercept_time:6.1f}s "
            f"over position ({x:,.0f} m, {y:,.0f} m)."
        )

    if result.intercept_time is not None and result.intercept_target_label == "decoy":
        x, y = result.intercept_position or (0.0, 0.0)
        interceptor_name = None
        for name, report in result.interceptor_reports.items():
            if report.target_label == "decoy" and report.intercept_time == result.intercept_time:
                interceptor_name = name
                break
        prefix = f"{interceptor_name} interceptor" if interceptor_name else "Interceptor"
        return (
            f"{prefix} collided with a decoy at t={result.intercept_time:6.1f}s "
            f"over position ({x:,.0f} m, {y:,.0f} m); primary warhead continued."
        )

    if result.icbm_impact_time is not None:
        return (
            "Interceptor failed to engage before impact. "
            f"ICBM reached ground at t={result.icbm_impact_time:6.1f}s."
        )

    return "Simulation ended without intercept or ground impact (timeout)."


def _describe_interceptor(name: str, report: InterceptorReport, icbm_impact_time: Optional[float]) -> str:
    if report.launch_time is None:
        return f"{name}: never launched (outside engagement window)"

    desc = f"{name}: launched t={report.launch_time:6.1f}s"
    if report.intercept_time is not None:
        target = report.target_label or "unknown"
        if report.success and target == "primary":
            outcome = "primary kill"
        elif target == "decoy":
            outcome = "decoy intercept"
        elif report.success:
            outcome = "successful intercept"
        else:
            outcome = "intercept"
        desc += f", {outcome} at t={report.intercept_time:6.1f}s"
    else:
        if report.expended:
            if icbm_impact_time is not None:
                desc += f", missed before impact at t={icbm_impact_time:6.1f}s"
            else:
                desc += ", expended without intercept"
        else:
            if icbm_impact_time is not None:
                desc += f", still in flight when impact occurred at t={icbm_impact_time:6.1f}s"
            else:
                desc += ", still active at simulation end"
    return desc


def run_monte_carlo(
    runs: int,
    *,
    seed: Optional[int] = None,
    base_kwargs: Optional[Dict[str, float]] = None,
    details: Optional[List[Dict[str, Any]]] = None,
) -> MonteCarloSummary:
    if runs <= 0:
        raise ValueError("runs must be positive")

    master_rng = random.Random(seed)
    base_kwargs = dict(base_kwargs or {})

    launch_delay_base = base_kwargs.get("interceptor_launch_delay", 120.0)
    speed_cap_base = base_kwargs.get("interceptor_speed_cap", 5000.0)
    noise_base = base_kwargs.get("guidance_noise_std_deg", 0.10)
    wind_base = base_kwargs.get("wind_velocity", (0.0, 0.0))
    gbi_salvo_base = max(1, int(base_kwargs.get("gbi_salvo_count", 1)))
    gbi_salvo_interval_base = max(0.0, base_kwargs.get("gbi_salvo_interval", 0.0))
    thaad_salvo_base = max(1, int(base_kwargs.get("thaad_salvo_count", 1)))
    thaad_salvo_interval_base = max(0.0, base_kwargs.get("thaad_salvo_interval", 0.0))
    boost_profile_base = tuple(base_kwargs.get("icbm_boost_profile", DEFAULT_BOOST_PROFILE))
    pitch_base = tuple(base_kwargs.get("icbm_pitch_schedule_deg", DEFAULT_PITCH_SCHEDULE_DEG))
    accel_base = base_kwargs.get("interceptor_max_accel", 60.0)
    decoy_release_base = base_kwargs.get("decoy_release_time", 220.0)
    decoy_count_base = int(base_kwargs.get("decoy_count", 3))
    decoy_spread_base = base_kwargs.get("decoy_spread_velocity", 280.0)
    decoy_confusion_base = base_kwargs.get("decoy_confusion_probability", 0.1)
    decoy_reacquire_base = base_kwargs.get("decoy_reacquisition_rate", 0.015)
    warhead_mass_base = base_kwargs.get("warhead_mass_fraction", 0.35)
    warhead_drag_base = base_kwargs.get("warhead_drag_multiplier", 0.6)
    decoy_mass_base = base_kwargs.get("decoy_mass_fraction", 0.04)
    decoy_drag_base = base_kwargs.get("decoy_drag_multiplier", 4.0)

    successes = impacts = timeouts = 0
    decoy_intercepts = 0
    intercept_times: List[float] = []
    miss_distances: List[float] = []
    layer_primary_counts: Dict[str, int] = {}
    layer_decoy_counts: Dict[str, int] = {}

    for run_index in range(runs):
        run_seed = master_rng.randint(0, 2**31 - 1)
        run_rng = random.Random(run_seed)

        kwargs = dict(base_kwargs)
        kwargs["rng"] = run_rng
        kwargs["interceptor_launch_delay"] = max(
            10.0, run_rng.gauss(launch_delay_base, max(5.0, 0.2 * abs(launch_delay_base)))
        )
        kwargs["interceptor_speed_cap"] = max(
            1500.0, run_rng.gauss(speed_cap_base, max(200.0, 0.15 * abs(speed_cap_base)))
        )
        kwargs["guidance_noise_std_deg"] = max(
            0.0, run_rng.gauss(noise_base, max(0.05, 0.3 * noise_base if noise_base else 0.1))
        )
        kwargs["gbi_salvo_count"] = gbi_salvo_base
        kwargs["gbi_salvo_interval"] = gbi_salvo_interval_base
        kwargs["thaad_salvo_count"] = thaad_salvo_base
        kwargs["thaad_salvo_interval"] = thaad_salvo_interval_base
        kwargs["wind_velocity"] = (
            run_rng.gauss(wind_base[0], 80.0),
            run_rng.gauss(wind_base[1], 12.0),
        )
        profile_variation: List[Tuple[float, float]] = []
        for idx, (duration, accel) in enumerate(boost_profile_base):
            dur_sigma = max(5.0, 0.15 * duration)
            acc_sigma = max(0.5, 0.25 * max(abs(accel), 1.0))
            duration_sample = max(5.0, run_rng.gauss(duration, dur_sigma))
            accel_sample = run_rng.gauss(accel, acc_sigma)
            profile_variation.append((duration_sample, accel_sample))
        kwargs["icbm_boost_profile"] = tuple(profile_variation)

        if pitch_base:
            pitch_samples: List[float] = []
            last_angle = pitch_base[-1]
            for idx in range(len(profile_variation)):
                base_angle = pitch_base[idx] if idx < len(pitch_base) else last_angle
                pitch_samples.append(run_rng.gauss(base_angle, 1.5))
            kwargs["icbm_pitch_schedule_deg"] = tuple(pitch_samples)

        kwargs["interceptor_max_accel"] = max(
            5.0, run_rng.gauss(accel_base, max(1.0, 0.2 * max(accel_base, 1.0)))
        )
        if decoy_release_base is None:
            kwargs["decoy_release_time"] = None
        else:
            kwargs["decoy_release_time"] = max(20.0, run_rng.gauss(decoy_release_base, 30.0))
        kwargs["decoy_count"] = max(0, int(round(run_rng.gauss(decoy_count_base, max(1.0, 0.5 * decoy_count_base)))))
        kwargs["decoy_spread_velocity"] = max(50.0, run_rng.gauss(decoy_spread_base, 0.25 * max(decoy_spread_base, 1.0)))
        kwargs["decoy_confusion_probability"] = min(
            0.4, max(0.0, run_rng.gauss(decoy_confusion_base, 0.08))
        )
        kwargs["decoy_reacquisition_rate"] = max(
            0.0, run_rng.gauss(decoy_reacquire_base, 0.5 * max(decoy_reacquire_base, 0.005))
        )
        kwargs["warhead_mass_fraction"] = min(
            0.9, max(0.05, run_rng.gauss(warhead_mass_base, 0.05))
        )
        kwargs["warhead_drag_multiplier"] = max(
            0.1, run_rng.gauss(warhead_drag_base, 0.1)
        )
        kwargs["decoy_mass_fraction"] = max(
            0.005, run_rng.gauss(decoy_mass_base, 0.01)
        )
        kwargs["decoy_drag_multiplier"] = max(
            0.5, run_rng.gauss(decoy_drag_base, 0.6)
        )

        result = simulate_icbm_intercept(**kwargs)

        if result.intercept_success and result.intercept_time is not None:
            successes += 1
            intercept_times.append(result.intercept_time)
        else:
            if result.icbm_impact_time is not None:
                impacts += 1
            else:
                timeouts += 1

        miss_distance = _minimum_miss_distance(result)
        if math.isfinite(miss_distance):
            miss_distances.append(miss_distance)

        for report in result.interceptor_reports.values():
            if report.target_label == "primary" and report.success:
                base = report.config_name
                layer_primary_counts[base] = layer_primary_counts.get(base, 0) + 1
            if report.target_label == "decoy":
                base = report.config_name
                layer_decoy_counts[base] = layer_decoy_counts.get(base, 0) + 1
                decoy_intercepts += 1

        if details is not None:
            record_kwargs = dict(kwargs)
            record_kwargs.pop("rng", None)
            entry = _result_to_entry(
                result,
                mode="monte_carlo",
                run_index=run_index,
                seed=run_seed,
                min_distance=miss_distance,
            )
            entry["drawn_parameters"] = record_kwargs
            details.append(entry)

    return MonteCarloSummary(
        runs=runs,
        successes=successes,
        impacts=impacts,
        timeouts=timeouts,
        intercept_times=intercept_times,
        miss_distances=miss_distances,
        decoy_intercepts=decoy_intercepts,
        layer_primary_kills=layer_primary_counts,
        layer_decoy_hits=layer_decoy_counts,
    )


def _plot(result: SimulationResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # Matplotlib is optional.
        print("matplotlib not available - skipping plot.")
        return

    icbm_x = [sample.icbm_position[0] for sample in result.samples]
    icbm_y = [sample.icbm_position[1] for sample in result.samples]

    interceptor_paths: Dict[str, List[Optional[Vector]]] = {
        name: [] for name in result.interceptor_reports.keys()
    }
    for sample in result.samples:
        for name in interceptor_paths.keys():
            interceptor_paths[name].append(sample.interceptor_positions_map.get(name))

    # Track decoy trajectories by unique ID so colors remain stable even if others are removed.
    decoy_id_order: List[int] = []
    for sample in result.samples:
        for decoy_id in sample.decoy_ids:
            if decoy_id not in decoy_id_order:
                decoy_id_order.append(decoy_id)

    decoy_paths: Dict[int, Tuple[List[float], List[float]]] = {}
    for decoy_id in decoy_id_order:
        decoy_paths[decoy_id] = ([], [])

    for sample in result.samples:
        id_to_pos = {
            decoy_id: pos for decoy_id, pos in zip(sample.decoy_ids, sample.decoy_positions)
        }
        for decoy_id in decoy_id_order:
            xs, ys = decoy_paths[decoy_id]
            pos = id_to_pos.get(decoy_id)
            if pos is None:
                xs.append(math.nan)
                ys.append(math.nan)
            else:
                xs.append(pos[0])
                ys.append(pos[1])

    plt.figure(figsize=(10, 5))
    plt.plot(icbm_x, icbm_y, color="#4D4D4D", linewidth=2.0, label="ICBM")

    for name, positions in interceptor_paths.items():
        filtered = [p for p in positions if p is not None]
        if not filtered:
            continue
        x_vals = [p[0] for p in filtered]
        y_vals = [p[1] for p in filtered]
        report = result.interceptor_reports[name]
        style = _interceptor_style(report.config_name)
        plt.plot(x_vals, y_vals, label=f"{name}", **style)

    if decoy_id_order:
        cmap = plt.get_cmap("tab10")
        for display_idx, decoy_id in enumerate(decoy_id_order, start=1):
            xs, ys = decoy_paths[decoy_id]
            if all(math.isnan(x) for x in xs):
                continue
            color = cmap((display_idx - 1) % cmap.N)
            plt.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=0.8,
                color=color,
                label=f"Decoy {display_idx}",
            )

    for name, report in result.interceptor_reports.items():
        if report.intercept_position is None:
            continue
        ix, iy = report.intercept_position
        style = _interceptor_style(report.config_name)
        marker_color = style.get("color", "tab:red")
        if report.target_label == "primary" and report.success:
            label = f"{name} primary hit"
            size = 110
        elif report.target_label == "decoy":
            label = f"{name} decoy intercept"
            size = 90
        else:
            label = f"{name} intercept"
            size = 90
        plt.scatter([ix], [iy], color=marker_color, marker="x", s=size, label=label)

    plt.axhline(0.0, color="black", linewidth=0.6)
    plt.xlabel("Range (m)")
    plt.ylabel("Altitude (m)")
    plt.title("ICBM Intercept Simulation")
    plt.legend()
    plt.grid(True, linewidth=0.2)
    plt.tight_layout()
    plt.show()


def _animate(result: SimulationResult, interval_ms: int = 35) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib animation support not available - skipping animation.")
        return

    icbm_x = [sample.icbm_position[0] for sample in result.samples]
    icbm_y = [sample.icbm_position[1] for sample in result.samples]

    interceptor_names = list(result.interceptor_reports.keys())
    interceptor_paths: Dict[str, Tuple[List[float], List[float]]] = {}
    for name in interceptor_names:
        xs: List[float] = []
        ys: List[float] = []
        for sample in result.samples:
            pos = sample.interceptor_positions_map.get(name)
            if pos is None:
                xs.append(math.nan)
                ys.append(math.nan)
            else:
                xs.append(pos[0])
                ys.append(pos[1])
        interceptor_paths[name] = (xs, ys)

    decoy_id_order: List[int] = []
    for sample in result.samples:
        for decoy_id in sample.decoy_ids:
            if decoy_id not in decoy_id_order:
                decoy_id_order.append(decoy_id)

    decoy_paths: Dict[int, Tuple[List[float], List[float]]] = {}
    for decoy_id in decoy_id_order:
        decoy_paths[decoy_id] = ([], [])

    for sample in result.samples:
        id_to_pos = {
            decoy_id: pos for decoy_id, pos in zip(sample.decoy_ids, sample.decoy_positions)
        }
        for decoy_id in decoy_id_order:
            xs, ys = decoy_paths[decoy_id]
            pos = id_to_pos.get(decoy_id)
            if pos is None:
                xs.append(math.nan)
                ys.append(math.nan)
            else:
                xs.append(pos[0])
                ys.append(pos[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("ICBM Intercept Simulation (Animation)")
    ax.grid(True, linewidth=0.2)

    icbm_line, = ax.plot([], [], color="#4D4D4D", linewidth=2.0, label="ICBM")
    interceptor_lines: Dict[str, any] = {}
    for name in interceptor_names:
        report = result.interceptor_reports[name]
        style = _interceptor_style(report.config_name)
        style.setdefault("linewidth", 2.0)
        interceptor_lines[name], = ax.plot([], [], label=f"{name}", **style)
    decoy_lines: Dict[int, any] = {}
    cmap = plt.get_cmap("tab10")
    for display_idx, decoy_id in enumerate(decoy_id_order, start=1):
        color = cmap((display_idx - 1) % cmap.N)
        decoy_lines[decoy_id], = ax.plot(
            [], [], linestyle="--", linewidth=0.7, color=color, label=f"Decoy {display_idx}"
        )

    intercept_markers = []
    for name, report in result.interceptor_reports.items():
        if report.intercept_position is None:
            continue
        style = _interceptor_style(report.config_name)
        color = style.get("color", "tab:red" if report.success else "orange")
        marker = ax.scatter(
            [report.intercept_position[0]],
            [report.intercept_position[1]],
            color=color,
            marker="x",
            s=100 if report.success else 80,
            label=f"{name} {'kill' if report.success else 'decoy'}",
        )
        intercept_markers.append(marker)

    ax.legend()

    def init() -> List[any]:
        icbm_line.set_data([], [])
        for line in interceptor_lines.values():
            line.set_data([], [])
        for line in decoy_lines.values():
            line.set_data([], [])
        return [icbm_line, *interceptor_lines.values(), *decoy_lines.values(), *intercept_markers]

    def update(frame: int) -> List[any]:
        upto = frame + 1
        icbm_line.set_data(icbm_x[:upto], icbm_y[:upto])
        for name, line in interceptor_lines.items():
            xs, ys = interceptor_paths[name]
            line.set_data(xs[:upto], ys[:upto])
        for decoy_id, line in decoy_lines.items():
            xs, ys = decoy_paths[decoy_id]
            line.set_data(xs[:upto], ys[:upto])
        return [icbm_line, *interceptor_lines.values(), *decoy_lines.values(), *intercept_markers]

    # Establish axis limits so trajectories are visible from the first frame.
    all_x: List[float] = [x for x in icbm_x]
    all_y: List[float] = [y for y in icbm_y]
    for xs, ys in interceptor_paths.values():
        all_x.extend(x for x in xs if not math.isnan(x))
        all_y.extend(y for y in ys if not math.isnan(y))
    for xs, ys in decoy_paths.values():
        all_x.extend(x for x in xs if not math.isnan(x))
        all_y.extend(y for y in ys if not math.isnan(y))
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        span_x = max(1.0, x_max - x_min)
        span_y = max(1.0, y_max - y_min)
        margin_x = 0.05 * span_x
        margin_y = 0.05 * span_y
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(max(0.0, y_min - margin_y), y_max + margin_y)

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(result.samples),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    # Keep a reference so Matplotlib does not garbage collect the animation before show().
    fig._animation = anim  # type: ignore[attr-defined]
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple ICBM intercept simulation.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="render a trajectory plot (requires matplotlib)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="play an animated visualization (requires matplotlib)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="number of simulations to run (use >1 for Monte Carlo mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible noisy guidance / Monte Carlo studies",
    )
    parser.add_argument(
        "--log-json",
        type=Path,
        default=None,
        help="write structured run data to a JSON lines file",
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="append to the JSON log instead of overwriting it",
    )
    parser.add_argument(
        "--gbi-salvo",
        type=int,
        default=1,
        help="number of exo-atmospheric interceptors launched per salvo (default 1)",
    )
    parser.add_argument(
        "--gbi-salvo-interval",
        type=float,
        default=0.0,
        help="spacing in seconds between exo-atmospheric salvo launches (default 0)",
    )
    parser.add_argument(
        "--thaad-salvo",
        type=int,
        default=1,
        help="number of terminal interceptors launched per salvo (default 1)",
    )
    parser.add_argument(
        "--thaad-salvo-interval",
        type=float,
        default=0.0,
        help="spacing in seconds between terminal salvo launches (default 0)",
    )
    args = parser.parse_args()

    base_rng = random.Random(args.seed) if args.seed is not None else None
    result = simulate_icbm_intercept(
        rng=base_rng,
        gbi_salvo_count=args.gbi_salvo,
        gbi_salvo_interval=args.gbi_salvo_interval,
        thaad_salvo_count=args.thaad_salvo,
        thaad_salvo_interval=args.thaad_salvo_interval,
    )
    print(_summarize(result))
    print(f"Sample count: {len(result.samples)} | Intercept success: {result.intercept_success}")
    for name in sorted(result.interceptor_reports.keys()):
        report = result.interceptor_reports[name]
        print("  " + _describe_interceptor(name, report, result.icbm_impact_time))

    log_entries: List[Dict[str, Any]] = []
    if args.log_json is not None:
        log_entries.append(
            _result_to_entry(
                result,
                mode="single",
                run_index=0,
                seed=args.seed,
                min_distance=_minimum_miss_distance(result),
            )
        )

    if args.runs > 1:
        summary_seed = args.seed if args.seed is not None else random.randrange(0, 2**32)
        details_list: Optional[List[Dict[str, Any]]] = [] if args.log_json is not None else None
        summary = run_monte_carlo(
            args.runs,
            seed=summary_seed,
            base_kwargs={
                "gbi_salvo_count": args.gbi_salvo,
                "gbi_salvo_interval": args.gbi_salvo_interval,
                "thaad_salvo_count": args.thaad_salvo,
                "thaad_salvo_interval": args.thaad_salvo_interval,
            },
            details=details_list,
        )
        print()
        print(f"Monte Carlo seed: {summary_seed}")
        print(summary.to_report())

        if details_list is not None:
            log_entries.extend(details_list)
            summary_entry: Dict[str, Any] = {
                "mode": "monte_carlo_summary",
                "seed": summary_seed,
                "runs": summary.runs,
                "successes": summary.successes,
                "impacts": summary.impacts,
                "timeouts": summary.timeouts,
                "decoy_intercepts": summary.decoy_intercepts,
                "layer_primary_kills": summary.layer_primary_kills,
                "layer_decoy_hits": summary.layer_decoy_hits,
            }
            if summary.intercept_times:
                summary_entry["avg_intercept_time"] = statistics.mean(summary.intercept_times)
                if len(summary.intercept_times) > 1:
                    summary_entry["std_intercept_time"] = statistics.pstdev(summary.intercept_times)
            if summary.miss_distances:
                summary_entry["median_miss_distance"] = statistics.median(summary.miss_distances)
            log_entries.append(summary_entry)

    if args.plot:
        if args.runs > 1:
            print("Plotting is only available for single-run simulations. Re-run with --runs 1.")
        else:
            _plot(result)

    if args.animate:
        if args.runs > 1:
            print("Animation is only available for single-run simulations. Re-run with --runs 1.")
        else:
            _animate(result)

    if args.log_json is not None and log_entries:
        log_path = args.log_json.expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append_log else "w"
        with log_path.open(mode, encoding="utf-8") as f:
            for entry in log_entries:
                json.dump(entry, f)
                f.write("\n")
        print(f"Wrote {len(log_entries)} entries to {log_path}")


if __name__ == "__main__":
    main()
