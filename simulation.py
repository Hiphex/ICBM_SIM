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
import concurrent.futures
import math
import os
from dataclasses import asdict, dataclass, field
from random import Random
from typing import Any, Callable, Dict, List, Optional, Tuple
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

STANDARD_SEA_LEVEL_DENSITY = 1.225  # kg/m^3
STANDARD_GAS_CONSTANT_AIR = 287.05  # J/(kg*K)
STANDARD_GRAVITY = 9.80665  # m/s^2
STANDARD_ATMOSPHERE_LAYERS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.0, 288.15, 101325.0, -0.0065),
    (11000.0, 216.65, 22632.06, 0.0),
    (20000.0, 216.65, 5474.889, 0.001),
    (32000.0, 228.65, 868.019, 0.0028),
    (47000.0, 270.65, 110.906, 0.0),
    (51000.0, 270.65, 66.9389, -0.0028),
    (71000.0, 214.65, 3.95642, -0.002),
    (84852.0, 186.946, 0.3734, 0.0),
)


def standard_atmosphere_density(altitude: float) -> float:
    """US Standard Atmosphere 1976 density model (valid through ~86 km)."""
    height = max(0.0, altitude)
    for idx, (base_alt, base_temp, base_pressure, lapse_rate) in enumerate(
        STANDARD_ATMOSPHERE_LAYERS
    ):
        next_alt = STANDARD_ATMOSPHERE_LAYERS[idx + 1][0] if idx + 1 < len(STANDARD_ATMOSPHERE_LAYERS) else None
        if next_alt is None or height <= next_alt:
            if lapse_rate == 0.0:
                pressure = base_pressure * math.exp(
                    -STANDARD_GRAVITY * (height - base_alt) / (STANDARD_GAS_CONSTANT_AIR * base_temp)
                )
                temperature = base_temp
            else:
                temperature = base_temp + lapse_rate * (height - base_alt)
                pressure = base_pressure * (base_temp / temperature) ** (
                    STANDARD_GRAVITY / (STANDARD_GAS_CONSTANT_AIR * lapse_rate)
                )
            return pressure / (STANDARD_GAS_CONSTANT_AIR * temperature)

    return STANDARD_SEA_LEVEL_DENSITY * math.exp(-height / 8500.0)


def air_density_factory(
    *,
    atmospheric_density_sea_level: float,
    atmospheric_scale_height: float,
    use_standard_atmosphere: bool,
) -> Callable[[float], float]:
    if use_standard_atmosphere:
        scale = atmospheric_density_sea_level / STANDARD_SEA_LEVEL_DENSITY

        def density(altitude: float) -> float:
            return scale * standard_atmosphere_density(altitude)

        return density

    def density(altitude: float) -> float:
        if altitude <= 0.0:
            return atmospheric_density_sea_level
        return atmospheric_density_sea_level * math.exp(-altitude / atmospheric_scale_height)

    return density


def gravity_at_altitude(altitude: float, gravity: float, earth_radius: float) -> float:
    """Compute gravity that falls off with altitude."""
    height = max(0.0, altitude)
    return gravity * (earth_radius / (earth_radius + height)) ** 2


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


# ============================================================================
# Radar Detection Functions
# ============================================================================


def radar_horizon_distance(radar_height: float, target_altitude: float, earth_radius: float) -> float:
    """
    Calculate the maximum line-of-sight distance from a radar to a target
    accounting for Earth's curvature.
    
    Uses the geometric horizon formula:
    d = sqrt(2 * R * h_radar) + sqrt(2 * R * h_target)
    
    Args:
        radar_height: Height of radar antenna above ground (meters)
        target_altitude: Altitude of target above ground (meters)
        earth_radius: Radius of the Earth (meters)
    
    Returns:
        Maximum line-of-sight distance in meters
    """
    # Distance to horizon from radar
    d_radar = math.sqrt(2.0 * earth_radius * max(0.0, radar_height))
    # Distance to horizon from target
    d_target = math.sqrt(2.0 * earth_radius * max(0.0, target_altitude))
    return d_radar + d_target


def radar_detection_probability(
    distance: float,
    target_rcs: float,
    radar_max_range: float,
    min_rcs_at_max_range: float,
) -> float:
    """
    Calculate the probability of detecting a target based on radar range equation.
    
    The radar equation shows received power falls off as 1/r^4. Detection probability
    is modeled as a sigmoid function of the signal-to-noise ratio.
    
    Args:
        distance: Distance from radar to target (meters)
        target_rcs: Radar cross section of target (m^2)
        radar_max_range: Maximum detection range for min_rcs_at_max_range (meters)
        min_rcs_at_max_range: Minimum detectable RCS at max range (m^2)
    
    Returns:
        Detection probability (0.0 to 1.0)
    """
    if distance <= 0.0:
        return 1.0
    if distance > radar_max_range * 2.0:
        return 0.0
    
    # Effective range based on RCS (larger RCS = detectable at longer range)
    # From radar equation: range proportional to RCS^(1/4)
    rcs_factor = (target_rcs / min_rcs_at_max_range) ** 0.25
    effective_max_range = radar_max_range * rcs_factor
    
    # Sigmoid probability centered at effective max range
    # P(detect) = 1 / (1 + exp((distance - effective_max_range) / scale))
    scale = effective_max_range * 0.1  # 10% of effective range for transition
    if scale <= 0.0:
        scale = 1000.0
    
    exponent = (distance - effective_max_range) / scale
    # Clamp to avoid overflow
    exponent = max(-20.0, min(20.0, exponent))
    
    return 1.0 / (1.0 + math.exp(exponent))


def is_target_visible_to_radar(
    radar_pos: Vector,
    radar_height: float,
    target_pos: Vector,
    earth_radius: float,
) -> Tuple[bool, float]:
    """
    Check if a target is visible to a radar (above the horizon) and return the distance.
    
    Args:
        radar_pos: (x, y) position of radar (y is altitude)
        radar_height: Additional antenna height above radar position
        target_pos: (x, y) position of target
        earth_radius: Radius of Earth
    
    Returns:
        Tuple of (is_visible, distance_to_target)
    """
    # Calculate distance between radar and target
    dx = target_pos[0] - radar_pos[0]
    dy = target_pos[1] - radar_pos[1]
    distance = math.sqrt(dx * dx + dy * dy)
    
    # Radar effective height is its y-coordinate (altitude) plus antenna height
    radar_total_height = radar_pos[1] + radar_height
    target_altitude = target_pos[1]
    
    # Calculate horizon distance
    horizon_dist = radar_horizon_distance(radar_total_height, target_altitude, earth_radius)
    
    # Target is visible if distance is less than horizon distance
    is_visible = distance <= horizon_dist
    
    return is_visible, distance


def update_radar_tracking(
    radar_state: "RadarState",
    icbm_states: Dict[str, "ICBMState"],
    time: float,
    dt: float,
    earth_radius: float,
    rng: Random,
) -> None:
    """
    Update radar tracking for all visible objects (ICBMs and decoys).
    
    This function:
    1. Checks visibility of all objects based on radar horizon
    2. Attempts detection based on RCS and range
    3. Updates tracked objects with noisy position/velocity measurements
    4. Manages track initiation and deletion
    
    Args:
        radar_state: The radar state to update
        icbm_states: Dictionary of ICBM states
        time: Current simulation time
        dt: Time step
        earth_radius: Earth radius for horizon calculation
        rng: Random number generator
    """
    cfg = radar_state.config
    radar_pos = cfg.position
    
    # Check if enough time has passed for radar update
    if time - radar_state.last_update_time < 1.0 / cfg.update_rate:
        return
    radar_state.last_update_time = time
    
    # Track which objects we've seen this update
    seen_objects: set = set()
    
    for icbm_name, icbm_state in icbm_states.items():
        if not icbm_state.launched or icbm_state.destroyed or icbm_state.impacted:
            continue
        
        # Check warhead visibility and detection
        is_visible, distance = is_target_visible_to_radar(
            radar_pos, cfg.antenna_height, icbm_state.position, earth_radius
        )
        
        if is_visible:
            # Get RCS from config
            rcs = icbm_state.config.rcs
            detect_prob = radar_detection_probability(
                distance, rcs, cfg.max_range, cfg.min_rcs_at_max_range
            )
            
            if rng.random() < detect_prob:
                # Detected! Update or create track
                object_id = icbm_name
                seen_objects.add(object_id)
                
                # Add measurement noise
                noisy_pos = (
                    icbm_state.position[0] + rng.gauss(0.0, cfg.position_noise_std),
                    icbm_state.position[1] + rng.gauss(0.0, cfg.position_noise_std),
                )
                noisy_vel = (
                    icbm_state.velocity[0] + rng.gauss(0.0, cfg.velocity_noise_std),
                    icbm_state.velocity[1] + rng.gauss(0.0, cfg.velocity_noise_std),
                )
                
                if object_id in radar_state.tracked_objects:
                    track = radar_state.tracked_objects[object_id]
                    track.position = noisy_pos
                    track.velocity = noisy_vel
                    track.detection_count += 1
                    track.last_update_time = time
                    if track.detection_count >= cfg.track_initiation_threshold:
                        track.track_established = True
                    # Store history for ballistic coefficient estimation
                    track.position_history.append((time, noisy_pos))
                    track.velocity_history.append((time, noisy_vel))
                    # Keep only last 20 samples
                    if len(track.position_history) > 20:
                        track.position_history.pop(0)
                        track.velocity_history.pop(0)
                else:
                    # Create new track
                    radar_state.tracked_objects[object_id] = TrackedObject(
                        object_id=object_id,
                        icbm_name=icbm_name,
                        is_decoy=False,
                        decoy_index=None,
                        position=noisy_pos,
                        velocity=noisy_vel,
                        detection_count=1,
                        last_update_time=time,
                        position_history=[(time, noisy_pos)],
                        velocity_history=[(time, noisy_vel)],
                    )
        
        # Check decoys if deployed
        decoy_state = icbm_state.decoy_state
        if decoy_state.deployed and decoy_state.positions:
            for idx, decoy_pos in enumerate(decoy_state.positions):
                if idx >= len(decoy_state.ids):
                    continue
                decoy_id = decoy_state.ids[idx]
                
                is_visible, distance = is_target_visible_to_radar(
                    radar_pos, cfg.antenna_height, decoy_pos, earth_radius
                )
                
                if is_visible:
                    # Get decoy RCS (if available, otherwise use warhead RCS)
                    if idx < len(decoy_state.rcs_values):
                        rcs = decoy_state.rcs_values[idx]
                    else:
                        rcs = icbm_state.config.rcs  # Decoys designed to mimic warhead RCS
                    
                    detect_prob = radar_detection_probability(
                        distance, rcs, cfg.max_range, cfg.min_rcs_at_max_range
                    )
                    
                    if rng.random() < detect_prob:
                        object_id = f"{icbm_name}-decoy-{decoy_id}"
                        seen_objects.add(object_id)
                        
                        decoy_vel = decoy_state.velocities[idx] if idx < len(decoy_state.velocities) else (0.0, 0.0)
                        
                        noisy_pos = (
                            decoy_pos[0] + rng.gauss(0.0, cfg.position_noise_std),
                            decoy_pos[1] + rng.gauss(0.0, cfg.position_noise_std),
                        )
                        noisy_vel = (
                            decoy_vel[0] + rng.gauss(0.0, cfg.velocity_noise_std),
                            decoy_vel[1] + rng.gauss(0.0, cfg.velocity_noise_std),
                        )
                        
                        if object_id in radar_state.tracked_objects:
                            track = radar_state.tracked_objects[object_id]
                            track.position = noisy_pos
                            track.velocity = noisy_vel
                            track.detection_count += 1
                            track.last_update_time = time
                            if track.detection_count >= cfg.track_initiation_threshold:
                                track.track_established = True
                            track.position_history.append((time, noisy_pos))
                            track.velocity_history.append((time, noisy_vel))
                            if len(track.position_history) > 20:
                                track.position_history.pop(0)
                                track.velocity_history.pop(0)
                        else:
                            radar_state.tracked_objects[object_id] = TrackedObject(
                                object_id=object_id,
                                icbm_name=icbm_name,
                                is_decoy=True,
                                decoy_index=idx,
                                position=noisy_pos,
                                velocity=noisy_vel,
                                detection_count=1,
                                last_update_time=time,
                                position_history=[(time, noisy_pos)],
                                velocity_history=[(time, noisy_vel)],
                            )
    
    # Remove stale tracks (not seen for more than 5 seconds)
    stale_threshold = 5.0
    stale_objects = [
        obj_id for obj_id, track in radar_state.tracked_objects.items()
        if time - track.last_update_time > stale_threshold
    ]
    for obj_id in stale_objects:
        del radar_state.tracked_objects[obj_id]


# ============================================================================
# Ballistic Coefficient Discrimination
# ============================================================================


# Altitude threshold for atmospheric discrimination (meters)
DISCRIMINATION_ALTITUDE_THRESHOLD = 100_000.0  # 100 km


def estimate_ballistic_coefficient(
    track: "TrackedObject",
    air_density_func: Callable[[float], float],
    gravity: float,
    earth_radius: float,
) -> Tuple[float, float]:
    """
    Estimate the ballistic coefficient of a tracked object from its trajectory.
    
    Ballistic coefficient B = m / (Cd * A) determines how much drag affects an object.
    Warheads have high B (heavy, streamlined), decoys have low B (light, high drag).
    
    We estimate B by measuring the deceleration due to drag and comparing to expected
    drag at the current altitude. This is most effective below 100km where air density
    is significant.
    
    Args:
        track: The tracked object with position/velocity history
        air_density_func: Function returning air density at altitude
        gravity: Surface gravity
        earth_radius: Earth radius
    
    Returns:
        Tuple of (estimated_B, variance) - variance indicates confidence
    """
    if len(track.velocity_history) < 3:
        return 0.0, float('inf')
    
    # Calculate acceleration from velocity differences
    accel_estimates = []
    
    for i in range(1, len(track.velocity_history)):
        t1, v1 = track.velocity_history[i - 1]
        t2, v2 = track.velocity_history[i]
        dt = t2 - t1
        if dt <= 0.0:
            continue
        
        # Total acceleration observed
        ax = (v2[0] - v1[0]) / dt
        ay = (v2[1] - v1[1]) / dt
        
        # Get position at midpoint for altitude
        if i < len(track.position_history):
            _, pos = track.position_history[i]
            altitude = pos[1]
        else:
            altitude = track.position[1]
        
        # Expected gravitational acceleration
        g_local = gravity * (earth_radius / (earth_radius + max(0.0, altitude))) ** 2
        
        # Remove gravity to get drag acceleration
        drag_ay = ay + g_local  # ay is negative due to gravity, so add g
        drag_ax = ax
        
        # Magnitude of drag acceleration
        drag_accel = math.sqrt(drag_ax * drag_ax + drag_ay * drag_ay)
        
        # Get velocity magnitude at this point
        speed = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
        
        # Get air density
        rho = air_density_func(altitude)
        
        if drag_accel > 0.1 and speed > 100.0 and rho > 1e-10:
            # From drag equation: a_drag = 0.5 * rho * v^2 * Cd * A / m = 0.5 * rho * v^2 / B
            # Therefore: B = 0.5 * rho * v^2 / a_drag
            estimated_B = 0.5 * rho * speed * speed / drag_accel
            # Sanity check: B should be positive and reasonable
            if 10.0 < estimated_B < 100000.0:
                accel_estimates.append(estimated_B)
    
    if not accel_estimates:
        return 0.0, float('inf')
    
    # Calculate mean and variance
    mean_B = sum(accel_estimates) / len(accel_estimates)
    if len(accel_estimates) > 1:
        variance = sum((b - mean_B) ** 2 for b in accel_estimates) / (len(accel_estimates) - 1)
    else:
        variance = float('inf')
    
    # Store in track for history
    track.accel_estimates = accel_estimates[-10:]  # Keep last 10
    
    return mean_B, variance


def calculate_warhead_probability(
    estimated_B: float,
    variance: float,
    expected_warhead_B: float,
    expected_decoy_B: float,
    altitude: float,
) -> float:
    """
    Calculate the probability that a tracked object is a warhead based on its
    estimated ballistic coefficient.
    
    Uses a Bayesian approach comparing estimated B to expected warhead and decoy B values.
    Confidence is lower at high altitudes where drag is negligible.
    
    Args:
        estimated_B: Estimated ballistic coefficient
        variance: Variance of the estimate
        expected_warhead_B: Expected B for a warhead (typically 5000-20000 kg/m^2)
        expected_decoy_B: Expected B for a decoy (typically 100-1000 kg/m^2)
        altitude: Current altitude (discrimination less reliable above 100km)
    
    Returns:
        P(warhead) from 0.0 to 1.0
    """
    # At high altitude, discrimination is poor - return 0.5 (uncertain)
    if altitude > DISCRIMINATION_ALTITUDE_THRESHOLD:
        # Gradual decrease in discrimination capability above 100km
        altitude_factor = 1.0 - min(1.0, (altitude - DISCRIMINATION_ALTITUDE_THRESHOLD) / 50000.0)
        if altitude_factor <= 0.0:
            return 0.5
    else:
        altitude_factor = 1.0
    
    # If we don't have a reliable estimate, return uncertain
    if estimated_B <= 0.0 or variance == float('inf'):
        return 0.5
    
    # Use Gaussian likelihood ratio
    # P(B | warhead) / P(B | decoy)
    
    # Standard deviation for expected B values (assume 30% uncertainty)
    warhead_std = expected_warhead_B * 0.3
    decoy_std = expected_decoy_B * 0.3
    
    # Also incorporate estimation uncertainty
    total_warhead_var = warhead_std * warhead_std + variance
    total_decoy_var = decoy_std * decoy_std + variance
    
    # Gaussian log-likelihoods
    warhead_diff = estimated_B - expected_warhead_B
    decoy_diff = estimated_B - expected_decoy_B
    
    # Avoid division by zero
    if total_warhead_var <= 0.0:
        total_warhead_var = 1.0
    if total_decoy_var <= 0.0:
        total_decoy_var = 1.0
    
    log_likelihood_warhead = -0.5 * (warhead_diff * warhead_diff / total_warhead_var + math.log(total_warhead_var))
    log_likelihood_decoy = -0.5 * (decoy_diff * decoy_diff / total_decoy_var + math.log(total_decoy_var))
    
    # Convert to probability using softmax
    log_ratio = log_likelihood_warhead - log_likelihood_decoy
    # Clamp to avoid overflow
    log_ratio = max(-20.0, min(20.0, log_ratio))
    
    p_warhead = 1.0 / (1.0 + math.exp(-log_ratio))
    
    # Apply altitude factor (less confidence at high altitude)
    # Move probability toward 0.5 based on altitude factor
    p_warhead = 0.5 + (p_warhead - 0.5) * altitude_factor
    
    return p_warhead


def update_discrimination(
    radar_state: "RadarState",
    icbm_states: Dict[str, "ICBMState"],
    air_density_func: Callable[[float], float],
    gravity: float,
    earth_radius: float,
) -> None:
    """
    Update discrimination estimates for all tracked objects.
    
    This function estimates the ballistic coefficient of each tracked object
    and calculates the probability that it is a warhead vs a decoy.
    
    Args:
        radar_state: Radar state with tracked objects
        icbm_states: Dictionary of ICBM states (for expected B values)
        air_density_func: Air density function
        gravity: Surface gravity
        earth_radius: Earth radius
    """
    for object_id, track in radar_state.tracked_objects.items():
        # Get expected B values from the parent ICBM config
        icbm_state = icbm_states.get(track.icbm_name)
        if icbm_state is None:
            continue
        
        cfg = icbm_state.config
        
        # Expected warhead B = mass * warhead_mass_fraction / (Cd * warhead_drag * A)
        warhead_mass = cfg.mass * cfg.warhead_mass_fraction
        warhead_drag = cfg.drag_coefficient * cfg.warhead_drag_multiplier
        warhead_area = cfg.reference_area * cfg.warhead_drag_multiplier
        if warhead_drag * warhead_area > 0:
            expected_warhead_B = warhead_mass / (warhead_drag * warhead_area)
        else:
            expected_warhead_B = cfg.warhead_ballistic_coeff
        
        # Expected decoy B = mass * decoy_mass_fraction / (Cd * decoy_drag * A)
        decoy_mass = cfg.mass * cfg.decoy_mass_fraction
        decoy_drag = cfg.drag_coefficient * cfg.decoy_drag_multiplier
        decoy_area = cfg.reference_area * cfg.decoy_drag_multiplier
        if decoy_drag * decoy_area > 0:
            expected_decoy_B = decoy_mass / (decoy_drag * decoy_area)
        else:
            expected_decoy_B = expected_warhead_B * 0.1  # Decoys have ~10% of warhead B
        
        # Estimate B from tracking data
        estimated_B, variance = estimate_ballistic_coefficient(
            track, air_density_func, gravity, earth_radius
        )
        
        track.estimated_ballistic_coeff = estimated_B
        track.ballistic_coeff_variance = variance
        
        # Calculate warhead probability
        altitude = track.position[1]
        track.warhead_probability = calculate_warhead_probability(
            estimated_B, variance, expected_warhead_B, expected_decoy_B, altitude
        )


@dataclass
class TrajectorySample:
    time: float
    icbm_position: Vector  # legacy: first ICBM position for backward compatibility
    icbm_velocity: Vector  # legacy: first ICBM velocity for backward compatibility
    interceptor_position: Optional[Vector]
    interceptor_velocity: Optional[Vector]
    decoy_positions: List[Vector]
    decoy_ids: List[int]
    interceptor_positions_map: Dict[str, Optional[Vector]] = field(default_factory=dict)
    interceptor_velocities_map: Dict[str, Optional[Vector]] = field(default_factory=dict)
    # Multi-ICBM support
    icbm_positions: Dict[str, Vector] = field(default_factory=dict)  # name -> position
    icbm_velocities: Dict[str, Vector] = field(default_factory=dict)  # name -> velocity
    icbm_destroyed: Dict[str, bool] = field(default_factory=dict)  # name -> destroyed
    decoy_positions_by_icbm: Dict[str, List[Vector]] = field(default_factory=dict)  # icbm_name -> positions
    decoy_ids_by_icbm: Dict[str, List[int]] = field(default_factory=dict)  # icbm_name -> ids


@dataclass
class SimulationResult:
    intercept_success: bool  # legacy: True if first/only ICBM destroyed
    intercept_time: Optional[float]  # legacy: first intercept time
    intercept_position: Optional[Vector]  # legacy: first intercept position
    icbm_impact_time: Optional[float]  # legacy: first ICBM impact time
    samples: List[TrajectorySample]
    intercept_target_label: Optional[str]
    decoy_intercepts: List[Tuple[float, Vector, str]]
    decoy_count: int
    parameters: Dict[str, Any]
    interceptor_reports: Dict[str, "InterceptorReport"]
    # Multi-ICBM support
    icbm_outcomes: Dict[str, "ICBMOutcome"] = field(default_factory=dict)  # name -> outcome
    overall_success: bool = True  # True if ALL ICBMs destroyed
    partial_success_count: int = 0  # how many ICBMs destroyed
    total_icbm_count: int = 1  # total ICBMs in simulation
    defense_sites: List["DefenseSiteConfig"] = field(default_factory=list)  # defense configuration used
    # Radar and discrimination stats
    radar_tracks_count: int = 0
    max_discrimination_confidence: float = 0.0


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
    site_name: str = ""  # which defense site this interceptor belongs to
    battery_name: str = ""  # which battery this interceptor belongs to
    launcher_index: int = 0  # which launcher in the battery
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
    selected_decoy_id: Optional[int] = None
    target_icbm_name: Optional[str] = None  # which ICBM this interceptor is targeting
    # Discrimination-based targeting
    tracked_object_id: Optional[str] = None  # ID of the tracked object we're targeting
    target_warhead_probability: float = 1.0  # P(warhead) of our target at assignment time
    awaiting_discrimination: bool = False  # True if waiting for better discrimination data


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
    target_icbm_name: Optional[str] = None  # which ICBM this interceptor targeted
    site_name: str = ""  # which defense site
    battery_name: str = ""  # which battery


@dataclass
class DecoyState:
    positions: List[Vector]
    velocities: List[Vector]
    masses: List[float]
    drag_coefficients: List[float]
    reference_areas: List[float]
    ids: List[int]
    next_id: int = 0
    deployed: bool = False
    released_count: int = 0
    # Radar cross section for each decoy (designed to mimic warhead RCS)
    rcs_values: List[float] = field(default_factory=list)
    # Ballistic coefficient for each decoy (mass / (Cd * A)) - lower than warhead
    ballistic_coeffs: List[float] = field(default_factory=list)


# ============================================================================
# Radar and Tracking System
# ============================================================================


@dataclass(frozen=True)
class RadarConfig:
    """Configuration for a ground-based radar sensor."""
    name: str
    position: Vector  # (x, altitude) position of the radar
    # Maximum detection range in meters (limited by power and sensitivity)
    max_range: float = 4000_000.0  # 4000 km
    # Minimum detectable RCS at max range (m^2)
    min_rcs_at_max_range: float = 0.001
    # Tracking update rate (Hz)
    update_rate: float = 10.0
    # Position measurement noise standard deviation (meters)
    position_noise_std: float = 50.0
    # Velocity measurement noise standard deviation (m/s)
    velocity_noise_std: float = 5.0
    # Number of consecutive detections needed for stable track
    track_initiation_threshold: int = 3
    # Radar antenna height above ground (for horizon calculation)
    antenna_height: float = 30.0


@dataclass
class TrackedObject:
    """Represents a tracked object (warhead or decoy) in the radar system."""
    object_id: str  # unique identifier (e.g., "ICBM-1" or "ICBM-1-decoy-0")
    icbm_name: str  # parent ICBM name
    is_decoy: bool
    decoy_index: Optional[int]  # index in DecoyState if is_decoy, else None
    # Measured/filtered state
    position: Vector
    velocity: Vector
    # Tracking quality
    detection_count: int = 0
    track_established: bool = False
    last_update_time: float = 0.0
    # Discrimination data
    estimated_ballistic_coeff: float = 0.0
    ballistic_coeff_variance: float = float('inf')
    warhead_probability: float = 0.5  # P(warhead) based on discrimination
    # History for ballistic coefficient estimation
    position_history: List[Tuple[float, Vector]] = field(default_factory=list)
    velocity_history: List[Tuple[float, Vector]] = field(default_factory=list)
    accel_estimates: List[float] = field(default_factory=list)


@dataclass
class RadarState:
    """Runtime state for a radar during simulation."""
    config: RadarConfig
    tracked_objects: Dict[str, TrackedObject] = field(default_factory=dict)
    last_update_time: float = 0.0


# ============================================================================
# Multi-ICBM Support
# ============================================================================


@dataclass(frozen=True)
class ICBMConfig:
    """Configuration for a single ICBM in a salvo attack."""
    name: str
    start_position: Vector = (0.0, 0.0)
    initial_velocity: Vector = (2400.0, 6200.0)
    launch_time: float = 0.0  # staggered launches relative to simulation start
    mass: float = 30_000.0
    drag_coefficient: float = 0.12
    reference_area: float = 3.5
    boost_profile: Tuple[Tuple[float, float], ...] = DEFAULT_BOOST_PROFILE
    pitch_schedule_deg: Tuple[float, ...] = DEFAULT_PITCH_SCHEDULE_DEG
    decoy_count: int = 3
    decoy_release_time: Optional[float] = 220.0
    decoy_spread_velocity: float = 280.0
    decoy_drag_multiplier: float = 4.0
    decoy_mass_fraction: float = 0.04
    warhead_mass_fraction: float = 0.35
    warhead_drag_multiplier: float = 0.6
    mirv_count: int = 1
    mirv_release_time: Optional[float] = None
    mirv_spread_velocity: float = 120.0
    # Radar cross section in square meters (warhead RCS is typically 0.01-0.1 m^2)
    rcs: float = 0.05
    # Expected ballistic coefficient for warhead (mass / (Cd * A)) in kg/m^2
    warhead_ballistic_coeff: float = 10000.0


@dataclass
class ICBMState:
    """Runtime state for a single ICBM during simulation."""
    config: ICBMConfig
    position: Vector
    velocity: Vector
    initial_heading: Vector
    active_mass: float
    active_drag_coefficient: float
    active_reference_area: float
    decoy_state: DecoyState
    cumulative_stage_times: List[float] = field(default_factory=list)
    stage_accels: List[float] = field(default_factory=list)
    cumulative_pitch_angles: List[float] = field(default_factory=list)
    stage_count: int = 0
    launched: bool = False
    destroyed: bool = False
    impacted: bool = False
    impact_time: Optional[float] = None
    destroyed_by: Optional[str] = None  # interceptor label that destroyed this ICBM
    mirv_deployed: bool = False


@dataclass
class ICBMOutcome:
    """Outcome for a single ICBM after simulation."""
    name: str
    destroyed: bool
    impacted: bool
    escaped: bool  # simulation ended without intercept or impact
    destroyed_by: Optional[str]
    impact_time: Optional[float]
    intercept_time: Optional[float]
    intercept_position: Optional[Vector]
    decoys_deployed: int


# ============================================================================
# ICBM Variant Presets
# ============================================================================

# Standard ICBM - baseline parameters
ICBM_STANDARD = ICBMConfig(
    name="Standard-ICBM",
    initial_velocity=(2400.0, 6200.0),
    mass=30_000.0,
    drag_coefficient=0.12,
    reference_area=3.5,
    boost_profile=DEFAULT_BOOST_PROFILE,
    decoy_count=3,
    decoy_release_time=220.0,
)

# Fast ICBM - higher velocity, lighter, fewer decoys
ICBM_FAST = ICBMConfig(
    name="Fast-ICBM",
    initial_velocity=(2800.0, 7000.0),
    mass=22_000.0,
    drag_coefficient=0.10,
    reference_area=2.8,
    boost_profile=((65.0, 18.0), (75.0, 10.0), (90.0, 4.0)),
    decoy_count=2,
    decoy_release_time=180.0,
)

# Heavy ICBM - more mass, slower, more decoys
ICBM_HEAVY = ICBMConfig(
    name="Heavy-ICBM",
    initial_velocity=(2200.0, 5800.0),
    mass=45_000.0,
    drag_coefficient=0.15,
    reference_area=4.5,
    boost_profile=((50.0, 20.0), (60.0, 12.0), (75.0, 5.0)),
    decoy_count=5,
    decoy_release_time=250.0,
)

# Decoy-Heavy ICBM - optimized for decoy deployment
ICBM_DECOY_HEAVY = ICBMConfig(
    name="Decoy-Heavy-ICBM",
    initial_velocity=(2300.0, 6000.0),
    mass=35_000.0,
    drag_coefficient=0.13,
    reference_area=3.8,
    boost_profile=DEFAULT_BOOST_PROFILE,
    decoy_count=8,
    decoy_release_time=200.0,
    decoy_spread_velocity=350.0,
    decoy_mass_fraction=0.06,
)

# Dictionary of available ICBM variants
ICBM_VARIANTS: Dict[str, ICBMConfig] = {
    "standard": ICBM_STANDARD,
    "fast": ICBM_FAST,
    "heavy": ICBM_HEAVY,
    "decoy-heavy": ICBM_DECOY_HEAVY,
}


def create_mixed_salvo(
    spacing: float = 50_000.0,
    launch_interval: float = 0.0,
    variants: Optional[List[str]] = None,
) -> List[ICBMConfig]:
    """Create a mixed salvo of ICBM variants.
    
    Args:
        spacing: Horizontal spacing between ICBM launch positions (meters)
        launch_interval: Time interval between launches (seconds)
        variants: List of variant names to include. If None, uses one of each type.
    
    Returns:
        List of ICBMConfig objects positioned and timed for launch.
    """
    if variants is None:
        variants = ["standard", "fast", "heavy", "decoy-heavy"]
    
    # Count occurrences of each variant type for unique naming
    variant_counts: Dict[str, int] = {}
    
    configs: List[ICBMConfig] = []
    for i, variant_name in enumerate(variants):
        base_config = ICBM_VARIANTS.get(variant_name.lower(), ICBM_STANDARD)
        
        # Generate unique name for duplicates
        base_name = base_config.name
        variant_key = variant_name.lower()
        variant_counts[variant_key] = variant_counts.get(variant_key, 0) + 1
        
        # Check if there are multiple of this variant
        total_of_type = sum(1 for v in variants if v.lower() == variant_key)
        if total_of_type > 1:
            name = f"{base_name}-{variant_counts[variant_key]}"
        else:
            name = base_name
        
        # Create a new config with updated position and launch time
        config = ICBMConfig(
            name=name,
            start_position=(i * spacing, 0.0),
            initial_velocity=base_config.initial_velocity,
            launch_time=i * launch_interval,
            mass=base_config.mass,
            drag_coefficient=base_config.drag_coefficient,
            reference_area=base_config.reference_area,
            boost_profile=base_config.boost_profile,
            pitch_schedule_deg=base_config.pitch_schedule_deg,
            decoy_count=base_config.decoy_count,
            decoy_release_time=base_config.decoy_release_time,
            decoy_spread_velocity=base_config.decoy_spread_velocity,
            decoy_drag_multiplier=base_config.decoy_drag_multiplier,
            decoy_mass_fraction=base_config.decoy_mass_fraction,
            warhead_mass_fraction=base_config.warhead_mass_fraction,
            warhead_drag_multiplier=base_config.warhead_drag_multiplier,
            mirv_count=base_config.mirv_count,
            mirv_release_time=base_config.mirv_release_time,
            mirv_spread_velocity=base_config.mirv_spread_velocity,
        )
        configs.append(config)
    
    return configs


# ============================================================================
# Multi-Site Defense Architecture
# ============================================================================


@dataclass(frozen=True)
class LauncherConfig:
    """Configuration for a single launcher within a battery."""
    interceptor_count: int = 4  # missiles per launcher
    reload_time: float = 0.0  # time between launches from same launcher (0 = no reload)


@dataclass(frozen=True)
class BatteryConfig:
    """Configuration for a battery containing multiple launchers."""
    name: str
    interceptor_template: InterceptorConfig  # base config for all interceptors from this battery
    launchers: Tuple[LauncherConfig, ...] = (LauncherConfig(),)  # default: 1 launcher with 4 interceptors


@dataclass(frozen=True)
class DefenseSiteConfig:
    """Configuration for a defense site containing multiple batteries."""
    name: str
    position: Vector
    batteries: Tuple[BatteryConfig, ...] = ()  # e.g., (GBI battery, THAAD battery)


@dataclass
class LauncherState:
    """Runtime state for a launcher."""
    config: LauncherConfig
    battery_name: str
    site_name: str
    remaining_interceptors: int
    last_launch_time: Optional[float] = None


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


def _interceptor_time_series(
    result: SimulationResult,
) -> Tuple[Dict[str, Tuple[List[float], List[float]]], Dict[str, Optional[int]]]:
    """Return interceptor position histories and intercept indices.

    Each interceptor is mapped to a pair of ``(xs, ys)`` lists matching the
    sample timeline. Positions that are undefined (for example prior to
    launch) are represented as ``math.nan`` so that downstream visualizers can
    keep a consistent frame count. The accompanying dictionary stores the
    index of the last sample at or before the recorded intercept time, which
    allows plots to truncate the line once an intercept occurs while the
    animation routines can still render the full timeline.
    """

    series: Dict[str, Tuple[List[float], List[float]]] = {}
    intercept_indices: Dict[str, Optional[int]] = {}

    for name, report in result.interceptor_reports.items():
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
        series[name] = (xs, ys)

        intercept_time = report.intercept_time
        if intercept_time is None:
            intercept_indices[name] = None
            continue

        last_index: Optional[int] = None
        for idx, sample in enumerate(result.samples):
            if sample.time > intercept_time:
                break
            x_val = xs[idx]
            y_val = ys[idx]
            if math.isnan(x_val) or math.isnan(y_val):
                continue
            last_index = idx
        intercept_indices[name] = last_index

    return series, intercept_indices


def _snapshot_decoys(decoy_positions: List[Vector]) -> List[Vector]:
    return [(pos[0], pos[1] if pos[1] >= 0.0 else 0.0) for pos in decoy_positions]


# advance_icbm_state: integrates missile/decoy dynamics and triggers decoy deployment.
# update_interceptor_states: manages interceptor launch gating, guidance, and intercept outcomes.
# collect_samples: records per-step trajectory snapshots for analysis and plotting.


def _deploy_decoys(
    *,
    time: float,
    decoy_release_time: Optional[float],
    decoy_count: int,
    decoy_spread_velocity: float,
    decoy_drag_multiplier: float,
    decoy_mass_fraction: float,
    warhead_mass_fraction: float,
    warhead_drag_multiplier: float,
    icbm_mass: float,
    icbm_drag_coefficient: float,
    icbm_reference_area: float,
    icbm_pos: Vector,
    icbm_vel: Vector,
    decoy_state: DecoyState,
    active_mass: float,
    active_drag_coefficient: float,
    active_reference_area: float,
    interceptor_states: List[InterceptorState],
    rng: Random,
    icbm_rcs: float = 0.05,  # RCS for decoys (designed to mimic warhead)
) -> Tuple[DecoyState, float, float, float]:
    if (
        decoy_state.deployed
        or decoy_release_time is None
        or decoy_count <= 0
        or time < decoy_release_time
    ):
        return decoy_state, active_mass, active_drag_coefficient, active_reference_area

    decoy_state.deployed = True

    active_mass = max(1.0, icbm_mass * max(warhead_mass_fraction, 0.05))
    active_drag_coefficient = icbm_drag_coefficient * warhead_drag_multiplier
    active_reference_area = icbm_reference_area * warhead_drag_multiplier

    decoy_state.positions = []
    decoy_state.velocities = []
    decoy_state.masses = []
    decoy_state.drag_coefficients = []
    decoy_state.reference_areas = []
    decoy_state.ids = []
    decoy_state.rcs_values = []
    decoy_state.ballistic_coeffs = []

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

        decoy_state.positions.append(icbm_pos)
        decoy_state.velocities.append(new_velocity)
        decoy_mass = max(1.0, icbm_mass * max(decoy_mass_fraction, 0.01))
        decoy_state.masses.append(decoy_mass)
        drag_multiplier = max(decoy_drag_multiplier, 0.1)
        decoy_drag = icbm_drag_coefficient * drag_multiplier
        decoy_area = icbm_reference_area * drag_multiplier
        decoy_state.drag_coefficients.append(decoy_drag)
        decoy_state.reference_areas.append(decoy_area)
        decoy_state.ids.append(decoy_state.next_id)
        decoy_state.next_id += 1
        
        # Decoys are designed to mimic warhead RCS (with some variation)
        decoy_rcs = icbm_rcs * rng.uniform(0.8, 1.2)
        decoy_state.rcs_values.append(decoy_rcs)
        
        # Calculate ballistic coefficient for this decoy: B = m / (Cd * A)
        if decoy_drag * decoy_area > 0:
            decoy_B = decoy_mass / (decoy_drag * decoy_area)
        else:
            decoy_B = 100.0  # Default low B for decoy
        decoy_state.ballistic_coeffs.append(decoy_B)

    decoy_state.released_count = len(decoy_state.positions)

    if decoy_state.positions:
        for state in interceptor_states:
            if state.expended:
                continue
            if state.launched and state.position is not None:
                confused = rng.random() < state.config.confusion_probability
            else:
                confused = False
            if confused:
                state.target_mode = "decoy"
                state.selected_decoy_id = rng.choice(decoy_state.ids)
            else:
                state.target_mode = "primary"
                state.selected_decoy_id = None

    return decoy_state, active_mass, active_drag_coefficient, active_reference_area


def _spawn_mirv_warheads(
    *,
    time: float,
    icbm_state: ICBMState,
    rng: Random,
    existing_names: Optional[set[str]] = None,
) -> List[Tuple[str, ICBMState]]:
    cfg = icbm_state.config
    if (
        cfg.mirv_count <= 1
        or cfg.mirv_release_time is None
        or icbm_state.mirv_deployed
        or not icbm_state.launched
    ):
        return []

    relative_time = time - cfg.launch_time
    if relative_time < cfg.mirv_release_time:
        return []

    icbm_state.mirv_deployed = True

    total_warhead_mass = cfg.mass * max(cfg.warhead_mass_fraction, 0.01)
    per_warhead_mass = max(1.0, total_warhead_mass / cfg.mirv_count)
    per_drag_coefficient = cfg.drag_coefficient * cfg.warhead_drag_multiplier
    per_reference_area = cfg.reference_area * cfg.warhead_drag_multiplier

    icbm_state.active_mass = per_warhead_mass
    icbm_state.active_drag_coefficient = per_drag_coefficient
    icbm_state.active_reference_area = per_reference_area

    spawned: List[Tuple[str, ICBMState]] = []
    name_set = existing_names if existing_names is not None else set()
    for idx in range(cfg.mirv_count - 1):
        orientation = normalize((rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)))
        if orientation == (0.0, 0.0):
            orientation = (1.0, 0.0)
        spread_speed = max(
            0.0, rng.gauss(cfg.mirv_spread_velocity, max(20.0, 0.25 * cfg.mirv_spread_velocity))
        )
        spread_velocity = mul(orientation, spread_speed)
        new_velocity = add(icbm_state.velocity, spread_velocity)

        warhead_name = f"{cfg.name}-MIRV-{idx + 1}"
        if warhead_name in name_set:
            suffix = 1
            candidate = f"{warhead_name}-{suffix}"
            while candidate in name_set:
                suffix += 1
                candidate = f"{warhead_name}-{suffix}"
            warhead_name = candidate

        warhead_config = ICBMConfig(
            name=warhead_name,
            start_position=icbm_state.position,
            initial_velocity=new_velocity,
            launch_time=time,
            mass=per_warhead_mass,
            drag_coefficient=per_drag_coefficient,
            reference_area=per_reference_area,
            boost_profile=(),
            pitch_schedule_deg=(),
            decoy_count=0,
            decoy_release_time=None,
            decoy_spread_velocity=0.0,
            decoy_drag_multiplier=cfg.decoy_drag_multiplier,
            decoy_mass_fraction=cfg.decoy_mass_fraction,
            warhead_mass_fraction=1.0,
            warhead_drag_multiplier=1.0,
            mirv_count=1,
            mirv_release_time=None,
            mirv_spread_velocity=cfg.mirv_spread_velocity,
        )
        warhead_state = _init_icbm_state(warhead_config)
        warhead_state.launched = True
        warhead_state.position = icbm_state.position
        warhead_state.velocity = new_velocity
        spawned.append((warhead_name, warhead_state))
        name_set.add(warhead_name)

    return spawned


def _advance_decoy_states(
    *,
    dt: float,
    gravity: float,
    earth_radius: float,
    wind_velocity: Vector,
    air_density: Callable[[float], float],
    decoy_state: DecoyState,
) -> None:
    for idx in range(len(decoy_state.positions)):
        decoy_velocity = decoy_state.velocities[idx]
        rel_air = sub(decoy_velocity, wind_velocity)
        rel_speed_decoy = length(rel_air)
        decoy_drag_vec = (0.0, 0.0)
        decoy_mass = decoy_state.masses[idx]
        if rel_speed_decoy > 0.0 and decoy_mass > 0.0:
            rho = air_density(decoy_state.positions[idx][1])
            drag_mag_decoy = (
                0.5
                * rho
                * rel_speed_decoy
                * rel_speed_decoy
                * decoy_state.drag_coefficients[idx]
                * decoy_state.reference_areas[idx]
                / decoy_mass
            )
            decoy_drag_vec = mul(normalize(rel_air), -drag_mag_decoy)

        local_gravity = gravity_at_altitude(decoy_state.positions[idx][1], gravity, earth_radius)
        total_decoy_acc = add((0.0, -local_gravity), decoy_drag_vec)
        decoy_velocity = add(decoy_velocity, mul(total_decoy_acc, dt))
        decoy_position = add(decoy_state.positions[idx], mul(decoy_velocity, dt))

        decoy_state.velocities[idx] = decoy_velocity
        decoy_state.positions[idx] = decoy_position


def _advance_icbm_state_core(
    *,
    time: float,
    dt: float,
    gravity: float,
    earth_radius: float,
    wind_velocity: Vector,
    air_density: Callable[[float], float],
    icbm_pos: Vector,
    icbm_vel: Vector,
    initial_heading: Vector,
    stage_count: int,
    cumulative_stage_times: List[float],
    stage_accels: List[float],
    cumulative_pitch_angles: List[float],
    active_mass: float,
    active_drag_coefficient: float,
    active_reference_area: float,
    icbm_mass: float,
    icbm_drag_coefficient: float,
    icbm_reference_area: float,
    decoy_release_time: Optional[float],
    decoy_count: int,
    decoy_spread_velocity: float,
    decoy_drag_multiplier: float,
    warhead_mass_fraction: float,
    warhead_drag_multiplier: float,
    decoy_mass_fraction: float,
    decoy_state: DecoyState,
    interceptor_states: List[InterceptorState],
    rng: Random,
    icbm_rcs: float = 0.05,
) -> Tuple[Vector, Vector, float, float, float, DecoyState]:
    decoy_state, active_mass, active_drag_coefficient, active_reference_area = _deploy_decoys(
        time=time,
        decoy_release_time=decoy_release_time,
        decoy_count=decoy_count,
        decoy_spread_velocity=decoy_spread_velocity,
        decoy_drag_multiplier=decoy_drag_multiplier,
        decoy_mass_fraction=decoy_mass_fraction,
        warhead_mass_fraction=warhead_mass_fraction,
        warhead_drag_multiplier=warhead_drag_multiplier,
        icbm_mass=icbm_mass,
        icbm_drag_coefficient=icbm_drag_coefficient,
        icbm_reference_area=icbm_reference_area,
        icbm_pos=icbm_pos,
        icbm_vel=icbm_vel,
        decoy_state=decoy_state,
        active_mass=active_mass,
        active_drag_coefficient=active_drag_coefficient,
        active_reference_area=active_reference_area,
        interceptor_states=interceptor_states,
        rng=rng,
        icbm_rcs=icbm_rcs,
    )

    local_gravity = gravity_at_altitude(icbm_pos[1], gravity, earth_radius)
    gravity_vec = (0.0, -local_gravity)
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

    _advance_decoy_states(
        dt=dt,
        gravity=gravity,
        earth_radius=earth_radius,
        wind_velocity=wind_velocity,
        air_density=air_density,
        decoy_state=decoy_state,
    )

    return (
        icbm_pos,
        icbm_vel,
        active_mass,
        active_drag_coefficient,
        active_reference_area,
        decoy_state,
    )


def advance_icbm_state(
    *,
    time: float,
    dt: float,
    gravity: float,
    earth_radius: float,
    wind_velocity: Vector,
    air_density: Callable[[float], float],
    icbm_pos: Vector,
    icbm_vel: Vector,
    initial_heading: Vector,
    stage_count: int,
    cumulative_stage_times: List[float],
    stage_accels: List[float],
    cumulative_pitch_angles: List[float],
    active_mass: float,
    active_drag_coefficient: float,
    active_reference_area: float,
    icbm_mass: float,
    icbm_drag_coefficient: float,
    icbm_reference_area: float,
    decoy_release_time: Optional[float],
    decoy_count: int,
    decoy_spread_velocity: float,
    decoy_drag_multiplier: float,
    warhead_mass_fraction: float,
    warhead_drag_multiplier: float,
    decoy_mass_fraction: float,
    decoy_state: DecoyState,
    interceptor_states: List[InterceptorState],
    rng: Random,
    icbm_rcs: float = 0.05,
) -> Tuple[Vector, Vector, float, float, float, DecoyState]:
    return _advance_icbm_state_core(
        time=time,
        dt=dt,
        gravity=gravity,
        earth_radius=earth_radius,
        wind_velocity=wind_velocity,
        air_density=air_density,
        icbm_pos=icbm_pos,
        icbm_vel=icbm_vel,
        initial_heading=initial_heading,
        stage_count=stage_count,
        cumulative_stage_times=cumulative_stage_times,
        stage_accels=stage_accels,
        cumulative_pitch_angles=cumulative_pitch_angles,
        active_mass=active_mass,
        active_drag_coefficient=active_drag_coefficient,
        active_reference_area=active_reference_area,
        icbm_mass=icbm_mass,
        icbm_drag_coefficient=icbm_drag_coefficient,
        icbm_reference_area=icbm_reference_area,
        decoy_release_time=decoy_release_time,
        decoy_count=decoy_count,
        decoy_spread_velocity=decoy_spread_velocity,
        decoy_drag_multiplier=decoy_drag_multiplier,
        warhead_mass_fraction=warhead_mass_fraction,
        warhead_drag_multiplier=warhead_drag_multiplier,
        decoy_mass_fraction=decoy_mass_fraction,
        decoy_state=decoy_state,
        interceptor_states=interceptor_states,
        rng=rng,
        icbm_rcs=icbm_rcs,
    )


def _estimate_time_to_intercept(
    interceptor_site: Vector,
    target_pos: Vector,
    target_vel: Vector,
    interceptor_speed_cap: float,
) -> float:
    """
    Estimate the time for an interceptor to reach a target.
    
    Uses a simple closing-speed model: the interceptor flies at speed_cap toward
    the target, while the target continues on its current velocity. We compute
    the component of target velocity toward/away from the interceptor and use
    that to estimate closing speed.
    
    Returns:
        Estimated time to intercept in seconds. Returns math.inf if intercept
        is not kinematically possible (target moving away faster than interceptor).
    """
    # Vector from interceptor site to target
    to_target = sub(target_pos, interceptor_site)
    distance = length(to_target)
    
    if distance <= 0.0:
        return 0.0
    
    # Unit vector toward target
    to_target_unit = normalize(to_target)
    
    # Target velocity component along the line of sight (positive = moving away)
    target_radial_vel = to_target_unit[0] * target_vel[0] + to_target_unit[1] * target_vel[1]
    
    # Closing speed: interceptor moving toward target at speed_cap, 
    # target moving away at target_radial_vel
    closing_speed = interceptor_speed_cap - target_radial_vel
    
    if closing_speed <= 0.0:
        # Target moving away faster than we can chase
        return math.inf
    
    return distance / closing_speed


def update_interceptor_states(
    *,
    time: float,
    dt: float,
    icbm_pos: Vector,
    icbm_vel: Vector,
    interceptor_states: List[InterceptorState],
    intercept_success: bool,
    intercept_time: Optional[float],
    intercept_position: Optional[Vector],
    intercept_target_label: Optional[str],
    decoy_intercepts: List[Tuple[float, Vector, str]],
    decoy_state: DecoyState,
    rng: Random,
) -> Tuple[bool, Optional[float], Optional[Vector], Optional[str], DecoyState]:
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
            dependency_states = [s for s in interceptor_states if s.config.name == cfg.depends_on]
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
        
        # Check if we can reach the target before max_flight_time expires
        # Use a safety margin (0.85) to account for guidance overhead
        time_to_intercept_ok = True
        if cfg.max_flight_time > 0.0:
            estimated_tti = _estimate_time_to_intercept(
                cfg.site, icbm_pos, icbm_vel, cfg.speed_cap
            )
            # Only launch if we can plausibly reach the target in time
            # Allow launch if estimated time is within 85% of max flight time
            if estimated_tti > cfg.max_flight_time * 0.85:
                time_to_intercept_ok = False
        
        if (
            time >= state.planned_launch_time
            and cfg.engage_altitude_min <= target_altitude <= cfg.engage_altitude_max
            and range_ok
            and time_to_intercept_ok
        ):
            state.launched = True
            state.position = cfg.site
            state.velocity = (0.0, 0.0)
            state.launch_time = time
            if decoy_state.deployed and decoy_state.positions and rng.random() < cfg.confusion_probability:
                state.target_mode = "decoy"
                state.selected_decoy_id = rng.choice(decoy_state.ids)
            else:
                state.target_mode = "primary"
                state.selected_decoy_id = None

    for state in interceptor_states:
        if not state.launched or state.expended or state.position is None or state.velocity is None:
            continue

        cfg = state.config

        if cfg.max_flight_time > 0.0 and state.launch_time is not None:
            if time - state.launch_time > cfg.max_flight_time:
                state.expended = True
                continue

        if decoy_state.deployed and state.target_mode == "decoy" and decoy_state.positions:
            reacquire_prob = (
                1.0 - math.exp(-cfg.reacquisition_rate * dt) if cfg.reacquisition_rate > 0.0 else 0.0
            )
            if rng.random() < reacquire_prob:
                state.target_mode = "primary"
                state.selected_decoy_id = None

        target_pos = icbm_pos
        if state.target_mode == "decoy" and decoy_state.positions:
            id_to_pos = dict(zip(decoy_state.ids, decoy_state.positions))
            decoy_id = state.selected_decoy_id
            if decoy_id in id_to_pos:
                target_pos = id_to_pos[decoy_id]
            elif decoy_state.ids:
                fallback_id = decoy_state.ids[-1]
                target_pos = id_to_pos[fallback_id]
                state.selected_decoy_id = fallback_id
            else:
                state.target_mode = "primary"
                state.selected_decoy_id = None
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
        if decoy_state.positions and state.target_mode == "decoy":
            for idx, decoy_position in enumerate(decoy_state.positions):
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
            target_pos = (
                decoy_state.positions[decoy_hit_index]
                if decoy_hit_index < len(decoy_state.positions)
                else state.position
            )
            decoy_intercepts.append((time, target_pos, state.label))
            state.success = False
            state.expended = True
            state.intercept_time = time
            state.intercept_position = target_pos
            state.intercept_target_label = "decoy"
            state.position = target_pos
            state.velocity = (0.0, 0.0)
            removed_decoy_id: Optional[int] = None
            if decoy_hit_index < len(decoy_state.ids):
                removed_decoy_id = decoy_state.ids[decoy_hit_index]
            if decoy_hit_index < len(decoy_state.positions):
                decoy_state.positions.pop(decoy_hit_index)
                decoy_state.velocities.pop(decoy_hit_index)
                decoy_state.masses.pop(decoy_hit_index)
                decoy_state.drag_coefficients.pop(decoy_hit_index)
                decoy_state.reference_areas.pop(decoy_hit_index)
                decoy_state.ids.pop(decoy_hit_index)
            if removed_decoy_id is not None:
                for other_state in interceptor_states:
                    if other_state.selected_decoy_id == removed_decoy_id:
                        other_state.selected_decoy_id = None
            break

    return intercept_success, intercept_time, intercept_position, intercept_target_label, decoy_state


def collect_samples(
    *,
    time: float,
    icbm_pos: Vector,
    icbm_vel: Vector,
    interceptor_states: List[InterceptorState],
    interceptors: Optional[List[InterceptorConfig]],
    decoy_state: DecoyState,
    samples: List[TrajectorySample],
) -> None:
    interceptor_positions_map = {state.label: state.position for state in interceptor_states}
    interceptor_velocities_map = {state.label: state.velocity for state in interceptor_states}

    default_interceptor_position = None
    default_interceptor_velocity = None
    if interceptors:
        default_label = next(
            (state.label for state in interceptor_states if state.config is interceptors[0]), None
        )
        if default_label is not None:
            default_interceptor_position = interceptor_positions_map.get(default_label)
            default_interceptor_velocity = interceptor_velocities_map.get(default_label)

    decoy_snapshot = _snapshot_decoys(decoy_state.positions)

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
            decoy_ids=list(decoy_state.ids),
        )
    )


# ============================================================================
# Multi-ICBM Helper Functions
# ============================================================================


def _init_icbm_state(config: ICBMConfig) -> ICBMState:
    """Initialize an ICBMState from an ICBMConfig."""
    initial_heading = normalize(config.initial_velocity)
    if initial_heading == (0.0, 0.0):
        initial_heading = (0.0, 1.0)

    boost_profile: Tuple[Tuple[float, float], ...] = tuple(
        (max(0.0, duration), accel) for duration, accel in config.boost_profile if duration > 0.0
    )
    cumulative_stage_times: List[float] = []
    stage_accels: List[float] = []
    total_duration = 0.0
    for duration, accel in boost_profile:
        total_duration += duration
        cumulative_stage_times.append(total_duration)
        stage_accels.append(accel)

    stage_count = len(boost_profile)
    pitch_schedule = list(config.pitch_schedule_deg[:stage_count])
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

    return ICBMState(
        config=config,
        position=config.start_position,
        velocity=config.initial_velocity,
        initial_heading=initial_heading,
        active_mass=config.mass,
        active_drag_coefficient=config.drag_coefficient,
        active_reference_area=config.reference_area,
        decoy_state=DecoyState(
            positions=[],
            velocities=[],
            masses=[],
            drag_coefficients=[],
            reference_areas=[],
            ids=[],
        ),
        cumulative_stage_times=cumulative_stage_times,
        stage_accels=stage_accels,
        cumulative_pitch_angles=cumulative_pitch_angles,
        stage_count=stage_count,
        launched=False,
    )


def _init_interceptors_from_sites(
    defense_sites: List[DefenseSiteConfig],
    launcher_states: List[LauncherState],
) -> List[InterceptorState]:
    """Initialize interceptor states from defense site configurations."""
    interceptor_states: List[InterceptorState] = []

    for site in defense_sites:
        for battery in site.batteries:
            cfg = battery.interceptor_template
            # Update the interceptor config with the site position
            site_cfg = InterceptorConfig(
                name=cfg.name,
                site=site.position,
                launch_delay=cfg.launch_delay,
                engage_altitude_min=cfg.engage_altitude_min,
                engage_altitude_max=cfg.engage_altitude_max,
                speed_cap=cfg.speed_cap,
                guidance_gain=cfg.guidance_gain,
                damping_gain=cfg.damping_gain,
                intercept_distance=cfg.intercept_distance,
                max_accel=cfg.max_accel,
                guidance_noise_std_deg=cfg.guidance_noise_std_deg,
                confusion_probability=cfg.confusion_probability,
                reacquisition_rate=cfg.reacquisition_rate,
                max_flight_time=cfg.max_flight_time,
                depends_on=cfg.depends_on,
                dependency_grace_period=cfg.dependency_grace_period,
                salvo_count=cfg.salvo_count,
                salvo_interval=cfg.salvo_interval,
                engage_range_min=cfg.engage_range_min,
                engage_range_max=cfg.engage_range_max,
            )

            for launcher_idx, launcher in enumerate(battery.launchers):
                # Create launcher state
                launcher_state = LauncherState(
                    config=launcher,
                    battery_name=battery.name,
                    site_name=site.name,
                    remaining_interceptors=launcher.interceptor_count,
                )
                launcher_states.append(launcher_state)

                # Create interceptor states for each missile in the launcher
                for interceptor_idx in range(launcher.interceptor_count):
                    label = f"{site.name}_{battery.name}_{cfg.name}#{interceptor_idx + 1}"
                    planned_launch_time = cfg.launch_delay + interceptor_idx * max(0.0, cfg.salvo_interval)

                    interceptor_states.append(
                        InterceptorState(
                            config=site_cfg,
                            salvo_index=interceptor_idx,
                            label=label,
                            planned_launch_time=planned_launch_time,
                            site_name=site.name,
                            battery_name=battery.name,
                            launcher_index=launcher_idx,
                        )
                    )

    return interceptor_states


def _advance_icbm_state_multi(
    *,
    time: float,
    dt: float,
    gravity: float,
    earth_radius: float,
    wind_velocity: Vector,
    air_density: Callable[[float], float],
    icbm_state: ICBMState,
    interceptor_states: List[InterceptorState],
    rng: Random,
) -> None:
    """Advance a single ICBM state by one time step (mutates icbm_state)."""
    if icbm_state.destroyed or icbm_state.impacted:
        return

    cfg = icbm_state.config

    # Check if ICBM should launch (for staggered launches)
    if not icbm_state.launched:
        if time >= cfg.launch_time:
            icbm_state.launched = True
        else:
            return

    # Adjust time relative to this ICBM's launch
    relative_time = time - cfg.launch_time

    icbm_pos, icbm_vel, active_mass, active_drag_coefficient, active_reference_area, decoy_state = _advance_icbm_state_core(
        time=relative_time,
        dt=dt,
        gravity=gravity,
        earth_radius=earth_radius,
        wind_velocity=wind_velocity,
        air_density=air_density,
        icbm_pos=icbm_state.position,
        icbm_vel=icbm_state.velocity,
        initial_heading=icbm_state.initial_heading,
        stage_count=icbm_state.stage_count,
        cumulative_stage_times=icbm_state.cumulative_stage_times,
        stage_accels=icbm_state.stage_accels,
        cumulative_pitch_angles=icbm_state.cumulative_pitch_angles,
        active_mass=icbm_state.active_mass,
        active_drag_coefficient=icbm_state.active_drag_coefficient,
        active_reference_area=icbm_state.active_reference_area,
        icbm_mass=cfg.mass,
        icbm_drag_coefficient=cfg.drag_coefficient,
        icbm_reference_area=cfg.reference_area,
        decoy_release_time=cfg.decoy_release_time,
        decoy_count=cfg.decoy_count,
        decoy_spread_velocity=cfg.decoy_spread_velocity,
        decoy_drag_multiplier=cfg.decoy_drag_multiplier,
        warhead_mass_fraction=cfg.warhead_mass_fraction,
        warhead_drag_multiplier=cfg.warhead_drag_multiplier,
        decoy_mass_fraction=cfg.decoy_mass_fraction,
        decoy_state=icbm_state.decoy_state,
        interceptor_states=[s for s in interceptor_states if s.target_icbm_name == cfg.name],
        rng=rng,
        icbm_rcs=cfg.rcs,
    )

    icbm_state.position = icbm_pos
    icbm_state.velocity = icbm_vel
    icbm_state.active_mass = active_mass
    icbm_state.active_drag_coefficient = active_drag_coefficient
    icbm_state.active_reference_area = active_reference_area
    icbm_state.decoy_state = decoy_state


def _select_target_icbm(
    interceptor_state: InterceptorState,
    icbm_states: Dict[str, ICBMState],
    rng: Random,
    all_interceptor_states: Optional[List[InterceptorState]] = None,
) -> Optional[str]:
    """Select which ICBM an interceptor should target. Returns ICBM name or None.
    
    Distributes interceptors across ICBMs - prioritizes untargeted ICBMs before
    assigning additional interceptors to already-targeted ones.
    """
    # Get active (not destroyed, not impacted) ICBMs that have launched
    active_icbms = [
        (name, state) for name, state in icbm_states.items()
        if state.launched and not state.destroyed and not state.impacted
    ]

    if not active_icbms:
        return None

    cfg = interceptor_state.config
    site_pos = cfg.site

    # Filter by engagement envelope
    eligible_icbms = []
    for name, state in active_icbms:
        altitude = state.position[1]
        horiz_distance = abs(state.position[0] - site_pos[0])

        altitude_ok = cfg.engage_altitude_min <= altitude <= cfg.engage_altitude_max
        range_ok = True
        if cfg.engage_range_min > 0.0 and horiz_distance < cfg.engage_range_min:
            range_ok = False
        if cfg.engage_range_max > 0.0 and horiz_distance > cfg.engage_range_max:
            range_ok = False

        if altitude_ok and range_ok:
            eligible_icbms.append((name, state, horiz_distance))

    if not eligible_icbms:
        return None

    # Count how many interceptors are already targeting each ICBM
    icbm_interceptor_counts: Dict[str, int] = {name: 0 for name, _, _ in eligible_icbms}
    if all_interceptor_states is not None:
        for other_state in all_interceptor_states:
            if other_state is interceptor_state:
                continue
            if other_state.target_icbm_name and other_state.target_icbm_name in icbm_interceptor_counts:
                if not other_state.expended:  # Only count active interceptors
                    icbm_interceptor_counts[other_state.target_icbm_name] += 1

    # Sort by: (1) fewest interceptors already assigned, (2) closest by distance
    eligible_icbms.sort(key=lambda x: (icbm_interceptor_counts[x[0]], x[2]))
    return eligible_icbms[0][0]


def _select_target_with_discrimination(
    interceptor_state: InterceptorState,
    icbm_name: str,
    icbm_state: "ICBMState",
    radar_state: Optional["RadarState"],
    rng: Random,
    all_interceptor_states: Optional[List[InterceptorState]] = None,
) -> Tuple[str, Optional[int], float]:
    """
    Select which specific object (warhead or decoy) to target based on discrimination data.
    
    Uses radar tracking to estimate which objects are most likely to be warheads
    and distributes interceptors to maximize P(kill).
    
    Args:
        interceptor_state: The interceptor making the selection
        icbm_name: Name of the ICBM cluster to target
        icbm_state: State of the target ICBM
        radar_state: Radar state with tracked objects and discrimination data
        rng: Random number generator
        all_interceptor_states: All interceptors for deconfliction
    
    Returns:
        Tuple of (target_mode, selected_decoy_id, warhead_probability)
        target_mode is "primary" or "decoy"
    """
    decoy_state = icbm_state.decoy_state
    
    # If no radar or no decoys deployed, target primary
    if radar_state is None or not decoy_state.deployed or not decoy_state.positions:
        return "primary", None, 1.0
    
    # Collect all tracked objects for this ICBM cluster
    tracked_candidates: List[Tuple[str, float, bool, Optional[int]]] = []
    # (object_id, warhead_probability, is_primary, decoy_index)
    
    # Check for warhead track
    warhead_track = radar_state.tracked_objects.get(icbm_name)
    if warhead_track and warhead_track.track_established:
        tracked_candidates.append((
            icbm_name,
            warhead_track.warhead_probability,
            True,
            None,
        ))
    else:
        # No established track on warhead - assume it's there with default P=0.5
        tracked_candidates.append((icbm_name, 0.5, True, None))
    
    # Check for decoy tracks
    for idx, decoy_id in enumerate(decoy_state.ids):
        object_id = f"{icbm_name}-decoy-{decoy_id}"
        decoy_track = radar_state.tracked_objects.get(object_id)
        if decoy_track and decoy_track.track_established:
            tracked_candidates.append((
                object_id,
                decoy_track.warhead_probability,
                False,
                idx,
            ))
        else:
            # No track - can't discriminate, assume low P
            tracked_candidates.append((object_id, 0.3, False, idx))
    
    # Count how many interceptors are already targeting each object
    object_interceptor_counts: Dict[str, int] = {obj_id: 0 for obj_id, _, _, _ in tracked_candidates}
    if all_interceptor_states is not None:
        for other_state in all_interceptor_states:
            if other_state is interceptor_state:
                continue
            if other_state.expended:
                continue
            if other_state.target_icbm_name != icbm_name:
                continue
            
            # Determine what this interceptor is targeting
            if other_state.target_mode == "primary":
                target_obj_id = icbm_name
            elif other_state.selected_decoy_id is not None:
                target_obj_id = f"{icbm_name}-decoy-{other_state.selected_decoy_id}"
            else:
                target_obj_id = icbm_name
            
            if target_obj_id in object_interceptor_counts:
                object_interceptor_counts[target_obj_id] += 1
    
    # Sort candidates by:
    # 1. Highest warhead probability (descending)
    # 2. Fewest interceptors already assigned (ascending)
    # This ensures we prioritize likely warheads but distribute interceptors
    
    def sort_key(candidate: Tuple[str, float, bool, Optional[int]]) -> Tuple[float, int]:
        obj_id, p_warhead, is_primary, _ = candidate
        # Negative P so higher P comes first
        # Add small bonus for primary (actual warhead position)
        primary_bonus = 0.1 if is_primary else 0.0
        return (-(p_warhead + primary_bonus), object_interceptor_counts.get(obj_id, 0))
    
    tracked_candidates.sort(key=sort_key)
    
    # Select the best candidate that doesn't have too many interceptors
    # Use a threshold based on salvo size
    max_interceptors_per_object = max(2, interceptor_state.config.salvo_count)
    
    for obj_id, p_warhead, is_primary, decoy_idx in tracked_candidates:
        if object_interceptor_counts.get(obj_id, 0) < max_interceptors_per_object:
            if is_primary:
                return "primary", None, p_warhead
            else:
                # Find the decoy ID from the index
                if decoy_idx is not None and decoy_idx < len(decoy_state.ids):
                    return "decoy", decoy_state.ids[decoy_idx], p_warhead
    
    # Fallback: target the highest probability object regardless of saturation
    best = tracked_candidates[0]
    if best[2]:  # is_primary
        return "primary", None, best[1]
    else:
        if best[3] is not None and best[3] < len(decoy_state.ids):
            return "decoy", decoy_state.ids[best[3]], best[1]
    
    # Final fallback: primary
    return "primary", None, 0.5


def _update_interceptor_states_multi(
    *,
    time: float,
    dt: float,
    icbm_states: Dict[str, ICBMState],
    interceptor_states: List[InterceptorState],
    decoy_intercepts: List[Tuple[float, Vector, str, str]],  # (time, pos, interceptor_label, icbm_name)
    rng: Random,
    launcher_states: List[LauncherState],
    radar_state: Optional["RadarState"] = None,
    use_discrimination: bool = True,
) -> None:
    """Update all interceptor states for multi-ICBM targeting (mutates states).
    
    When use_discrimination is True and radar_state is provided, uses physics-based
    discrimination to select targets based on estimated ballistic coefficients.
    Otherwise falls back to probabilistic confusion model.
    """

    # Phase 1: Launch gating and target assignment
    for state in interceptor_states:
        if state.launched or state.expended:
            continue

        cfg = state.config

        # Check if any ICBM has been destroyed by this interceptor's layer
        any_success = any(
            s.success and s.config.name == cfg.name
            for s in interceptor_states
        )
        if any_success and cfg.depends_on is None:
            # This layer already achieved a kill; don't launch more
            pass  # Could still launch for other ICBMs

        # Check dependency
        dependency_ready = True
        if cfg.depends_on:
            dependency_states = [s for s in interceptor_states if s.config.name == cfg.depends_on]
            if dependency_states:
                # Check if dependency layer has failed or grace period exceeded
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

        if not dependency_ready:
            continue

        # Check timing
        if time < state.planned_launch_time:
            continue

        # Select target ICBM if not already assigned
        if state.target_icbm_name is None:
            target_name = _select_target_icbm(state, icbm_states, rng, interceptor_states)
            if target_name is None:
                continue
            state.target_icbm_name = target_name

        # Verify target is still valid
        target_icbm = icbm_states.get(state.target_icbm_name)
        if target_icbm is None or target_icbm.destroyed or target_icbm.impacted or not target_icbm.launched:
            # Re-select target
            state.target_icbm_name = None
            target_name = _select_target_icbm(state, icbm_states, rng, interceptor_states)
            if target_name is None:
                continue
            state.target_icbm_name = target_name
            target_icbm = icbm_states[target_name]

        # Check engagement envelope for selected target
        target_altitude = target_icbm.position[1]
        horiz_distance = abs(target_icbm.position[0] - cfg.site[0])

        altitude_ok = cfg.engage_altitude_min <= target_altitude <= cfg.engage_altitude_max
        range_ok = True
        if cfg.engage_range_min > 0.0 and horiz_distance < cfg.engage_range_min:
            range_ok = False
        if cfg.engage_range_max > 0.0 and horiz_distance > cfg.engage_range_max:
            range_ok = False

        # Check if we can reach the target before max_flight_time expires
        # Use a safety margin (0.85) to account for guidance overhead
        time_to_intercept_ok = True
        if cfg.max_flight_time > 0.0:
            estimated_tti = _estimate_time_to_intercept(
                cfg.site, target_icbm.position, target_icbm.velocity, cfg.speed_cap
            )
            # Only launch if we can plausibly reach the target in time
            if estimated_tti > cfg.max_flight_time * 0.85:
                time_to_intercept_ok = False

        if altitude_ok and range_ok and time_to_intercept_ok:
            state.launched = True
            state.position = cfg.site
            state.velocity = (0.0, 0.0)
            state.launch_time = time

            # Target selection: use discrimination if available, otherwise probabilistic
            decoy_state = target_icbm.decoy_state
            if use_discrimination and radar_state is not None:
                # Use physics-based discrimination
                target_mode, selected_decoy_id, warhead_prob = _select_target_with_discrimination(
                    state, state.target_icbm_name, target_icbm, radar_state, rng, interceptor_states
                )
                state.target_mode = target_mode
                state.selected_decoy_id = selected_decoy_id
                state.target_warhead_probability = warhead_prob
                state.tracked_object_id = (
                    state.target_icbm_name if target_mode == "primary"
                    else f"{state.target_icbm_name}-decoy-{selected_decoy_id}"
                )
            else:
                # Fallback to probabilistic confusion model
                if decoy_state.deployed and decoy_state.positions and rng.random() < cfg.confusion_probability:
                    state.target_mode = "decoy"
                    state.selected_decoy_id = rng.choice(decoy_state.ids)
                    state.target_warhead_probability = 0.0  # We know we're confused
                else:
                    state.target_mode = "primary"
                    state.selected_decoy_id = None
                    state.target_warhead_probability = 1.0

    # Phase 2: Guidance and intercept detection
    for state in interceptor_states:
        if not state.launched or state.expended or state.position is None or state.velocity is None:
            continue

        cfg = state.config

        # Check flight time limit
        if cfg.max_flight_time > 0.0 and state.launch_time is not None:
            if time - state.launch_time > cfg.max_flight_time:
                state.expended = True
                continue

        # Get target ICBM
        if state.target_icbm_name is None:
            state.expended = True
            continue

        target_icbm = icbm_states.get(state.target_icbm_name)
        if target_icbm is None or target_icbm.destroyed:
            # Target destroyed; try to reassign
            new_target = _select_target_icbm(state, icbm_states, rng, interceptor_states)
            if new_target is None:
                state.expended = True
                continue
            state.target_icbm_name = new_target
            state.target_mode = "primary"
            state.selected_decoy_id = None
            target_icbm = icbm_states[new_target]

        decoy_state = target_icbm.decoy_state

        # Reacquisition / re-discrimination during flight
        if use_discrimination and radar_state is not None:
            # With discrimination, continuously re-evaluate target based on updated probabilities
            # This allows the interceptor to switch if we realize we're tracking a decoy
            if decoy_state.deployed and decoy_state.positions:
                # Get current track's warhead probability
                current_p = state.target_warhead_probability
                
                # Check if there's a better target (higher P_warhead)
                best_mode, best_decoy_id, best_p = _select_target_with_discrimination(
                    state, state.target_icbm_name, target_icbm, radar_state, rng, interceptor_states
                )
                
                # Switch to better target if significantly more likely to be warhead
                # Use a threshold to avoid constant switching
                if best_p > current_p + 0.2:  # 20% improvement threshold
                    state.target_mode = best_mode
                    state.selected_decoy_id = best_decoy_id
                    state.target_warhead_probability = best_p
                    state.tracked_object_id = (
                        state.target_icbm_name if best_mode == "primary"
                        else f"{state.target_icbm_name}-decoy-{best_decoy_id}"
                    )
        else:
            # Fallback to probabilistic reacquisition
            if decoy_state.deployed and state.target_mode == "decoy" and decoy_state.positions:
                reacquire_prob = (
                    1.0 - math.exp(-cfg.reacquisition_rate * dt) if cfg.reacquisition_rate > 0.0 else 0.0
                )
                if rng.random() < reacquire_prob:
                    state.target_mode = "primary"
                    state.selected_decoy_id = None
                    state.target_warhead_probability = 1.0

        # Determine guidance target
        target_pos = target_icbm.position
        if state.target_mode == "decoy" and decoy_state.positions:
            id_to_pos = dict(zip(decoy_state.ids, decoy_state.positions))
            decoy_id = state.selected_decoy_id
            if decoy_id in id_to_pos:
                target_pos = id_to_pos[decoy_id]
            elif decoy_state.ids:
                fallback_id = decoy_state.ids[-1]
                target_pos = id_to_pos[fallback_id]
                state.selected_decoy_id = fallback_id
            else:
                state.target_mode = "primary"
                state.selected_decoy_id = None
                target_pos = target_icbm.position

        # Guidance
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

        # Intercept detection
        primary_distance = length(sub(target_icbm.position, state.position))
        decoy_hit_index: Optional[int] = None
        if decoy_state.positions and state.target_mode == "decoy":
            for idx, decoy_position in enumerate(decoy_state.positions):
                if length(sub(decoy_position, state.position)) <= cfg.intercept_distance:
                    decoy_hit_index = idx
                    break

        if primary_distance <= cfg.intercept_distance:
            # Primary kill
            state.success = True
            state.expended = True
            state.intercept_time = time
            state.intercept_position = target_icbm.position
            state.intercept_target_label = "primary"
            state.position = target_icbm.position
            state.velocity = (0.0, 0.0)

            target_icbm.destroyed = True
            target_icbm.destroyed_by = state.label
            continue

        if decoy_hit_index is not None:
            # Decoy intercept
            target_pos = (
                decoy_state.positions[decoy_hit_index]
                if decoy_hit_index < len(decoy_state.positions)
                else state.position
            )
            decoy_intercepts.append((time, target_pos, state.label, state.target_icbm_name or ""))
            state.success = False
            state.expended = True
            state.intercept_time = time
            state.intercept_position = target_pos
            state.intercept_target_label = "decoy"
            state.position = target_pos
            state.velocity = (0.0, 0.0)

            # Remove decoy
            removed_decoy_id: Optional[int] = None
            if decoy_hit_index < len(decoy_state.ids):
                removed_decoy_id = decoy_state.ids[decoy_hit_index]
            if decoy_hit_index < len(decoy_state.positions):
                decoy_state.positions.pop(decoy_hit_index)
                decoy_state.velocities.pop(decoy_hit_index)
                decoy_state.masses.pop(decoy_hit_index)
                decoy_state.drag_coefficients.pop(decoy_hit_index)
                decoy_state.reference_areas.pop(decoy_hit_index)
                decoy_state.ids.pop(decoy_hit_index)
            if removed_decoy_id is not None:
                for other_state in interceptor_states:
                    if other_state.selected_decoy_id == removed_decoy_id:
                        other_state.selected_decoy_id = None


def _collect_samples_multi(
    *,
    time: float,
    icbm_states: Dict[str, ICBMState],
    interceptor_states: List[InterceptorState],
    samples: List[TrajectorySample],
) -> None:
    """Collect trajectory samples for multi-ICBM simulation."""
    interceptor_positions_map = {state.label: state.position for state in interceptor_states}
    interceptor_velocities_map = {state.label: state.velocity for state in interceptor_states}

    # Build multi-ICBM data
    icbm_positions: Dict[str, Vector] = {}
    icbm_velocities: Dict[str, Vector] = {}
    icbm_destroyed: Dict[str, bool] = {}
    decoy_positions_by_icbm: Dict[str, List[Vector]] = {}
    decoy_ids_by_icbm: Dict[str, List[int]] = {}
    all_decoy_positions: List[Vector] = []
    all_decoy_ids: List[int] = []

    for name, state in icbm_states.items():
        icbm_positions[name] = state.position
        icbm_velocities[name] = state.velocity
        icbm_destroyed[name] = state.destroyed

        decoy_snapshot = _snapshot_decoys(state.decoy_state.positions)
        decoy_positions_by_icbm[name] = decoy_snapshot
        decoy_ids_by_icbm[name] = list(state.decoy_state.ids)
        all_decoy_positions.extend(decoy_snapshot)
        all_decoy_ids.extend(state.decoy_state.ids)

    # Legacy fields: use first ICBM
    first_icbm_name = next(iter(icbm_states.keys())) if icbm_states else None
    if first_icbm_name:
        first_state = icbm_states[first_icbm_name]
        legacy_icbm_position = first_state.position
        legacy_icbm_velocity = first_state.velocity
    else:
        legacy_icbm_position = (0.0, 0.0)
        legacy_icbm_velocity = (0.0, 0.0)

    # Legacy interceptor fields
    default_interceptor_position = None
    default_interceptor_velocity = None
    if interceptor_states:
        first_state = interceptor_states[0]
        default_interceptor_position = first_state.position
        default_interceptor_velocity = first_state.velocity

    samples.append(
        TrajectorySample(
            time,
            icbm_position=legacy_icbm_position,
            icbm_velocity=legacy_icbm_velocity,
            interceptor_position=default_interceptor_position,
            interceptor_velocity=default_interceptor_velocity,
            interceptor_positions_map=interceptor_positions_map,
            interceptor_velocities_map=interceptor_velocities_map,
            decoy_positions=all_decoy_positions,
            decoy_ids=all_decoy_ids,
            icbm_positions=icbm_positions,
            icbm_velocities=icbm_velocities,
            icbm_destroyed=icbm_destroyed,
            decoy_positions_by_icbm=decoy_positions_by_icbm,
            decoy_ids_by_icbm=decoy_ids_by_icbm,
        )
    )


def _json_safe(value: Any) -> Any:
    """Convert simulation parameters to JSON-friendly structures."""
    if isinstance(value, InterceptorConfig):
        value = asdict(value)

    if isinstance(value, dict):
        return {key: _json_safe(sub_value) for key, sub_value in value.items()}

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def _simulate_multi_icbm(
    *,
    dt: float,
    max_time: Optional[float],
    gravity: float,
    earth_radius: float,
    atmospheric_density_sea_level: float,
    atmospheric_scale_height: float,
    use_standard_atmosphere: bool,
    adaptive_time_step: bool,
    adaptive_dt_min: Optional[float],
    adaptive_dt_max: Optional[float],
    wind_velocity: Vector,
    rng: Random,
    icbm_configs: List[ICBMConfig],
    defense_sites: Optional[List[DefenseSiteConfig]],
    interceptors: Optional[List[InterceptorConfig]],
    interceptor_site: Vector,
    interceptor_speed_cap: float,
    interceptor_launch_delay: float,
    gbi_salvo_count: int,
    gbi_salvo_interval: float,
    thaad_salvo_count: int,
    thaad_salvo_interval: float,
    guidance_gain: float,
    damping_gain: float,
    intercept_distance: float,
    guidance_noise_std_deg: float,
    interceptor_max_accel: float,
    decoy_confusion_probability: float,
    decoy_reacquisition_rate: float,
    icbm_rcs_override: Optional[float] = None,
    warhead_ballistic_coeff_override: Optional[float] = None,
    # Radar and discrimination parameters
    use_discrimination: bool = True,
    radar_config: Optional[RadarConfig] = None,
    # Layer-specific interceptor overrides
    gbi_speed_cap_override: Optional[float] = None,
    thaad_speed_cap_override: Optional[float] = None,
    gbi_launch_delay_override: Optional[float] = None,
    thaad_launch_delay_override: Optional[float] = None,
    gbi_guidance_gain_override: Optional[float] = None,
    thaad_guidance_gain_override: Optional[float] = None,
    gbi_damping_gain_override: Optional[float] = None,
    thaad_damping_gain_override: Optional[float] = None,
    gbi_intercept_distance_override: Optional[float] = None,
    thaad_intercept_distance_override: Optional[float] = None,
    gbi_max_accel_override: Optional[float] = None,
    thaad_max_accel_override: Optional[float] = None,
    gbi_guidance_noise_override: Optional[float] = None,
    thaad_guidance_noise_override: Optional[float] = None,
    gbi_reacquisition_rate_override: Optional[float] = None,
    thaad_reacquisition_rate_override: Optional[float] = None,
) -> "SimulationResult":
    """Internal function to simulate multiple ICBMs with multi-site defense.
    
    When use_discrimination is True, creates a ground-based radar and uses
    physics-based ballistic coefficient discrimination for target selection.
    """

    # Initialize ICBM states
    icbm_states: Dict[str, ICBMState] = {}
    for cfg in icbm_configs:
        icbm_states[cfg.name] = _init_icbm_state(cfg)

    # Initialize defense
    launcher_states: List[LauncherState] = []
    interceptor_states: List[InterceptorState] = []

    if defense_sites is not None and len(defense_sites) > 0:
        # Use defense site configuration
        interceptor_states = _init_interceptors_from_sites(defense_sites, launcher_states)
    elif interceptors is not None:
        # Use legacy interceptors list
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
    else:
        # Create default interceptors using overrides if provided
        gbi_speed = gbi_speed_cap_override if gbi_speed_cap_override is not None else interceptor_speed_cap
        gbi_delay = gbi_launch_delay_override if gbi_launch_delay_override is not None else interceptor_launch_delay
        gbi_gain = gbi_guidance_gain_override if gbi_guidance_gain_override is not None else max(0.88, guidance_gain * 1.18)
        gbi_damping = gbi_damping_gain_override if gbi_damping_gain_override is not None else max(0.055, damping_gain * 1.15)
        gbi_dist = gbi_intercept_distance_override if gbi_intercept_distance_override is not None else max(intercept_distance, 96_000.0)
        gbi_accel = gbi_max_accel_override if gbi_max_accel_override is not None else interceptor_max_accel
        gbi_noise = gbi_guidance_noise_override if gbi_guidance_noise_override is not None else max(0.03, guidance_noise_std_deg * 0.9)
        gbi_reacq = gbi_reacquisition_rate_override if gbi_reacquisition_rate_override is not None else max(decoy_reacquisition_rate, 0.018)

        thaad_speed = thaad_speed_cap_override if thaad_speed_cap_override is not None else 5000.0
        thaad_delay = thaad_launch_delay_override if thaad_launch_delay_override is not None else interceptor_launch_delay + 220.0
        thaad_gain = thaad_guidance_gain_override if thaad_guidance_gain_override is not None else max(0.68, guidance_gain * 1.38)
        thaad_damping = thaad_damping_gain_override if thaad_damping_gain_override is not None else max(0.11, damping_gain * 1.9)
        thaad_dist = thaad_intercept_distance_override if thaad_intercept_distance_override is not None else 180_000.0
        thaad_accel = thaad_max_accel_override if thaad_max_accel_override is not None else max(interceptor_max_accel, 155.0)
        thaad_noise = thaad_guidance_noise_override if thaad_guidance_noise_override is not None else max(0.035, guidance_noise_std_deg * 1.05)
        thaad_reacq = thaad_reacquisition_rate_override if thaad_reacquisition_rate_override is not None else max(decoy_reacquisition_rate * 2.0, 0.06)

        normalized_gbi_salvo_count = max(1, int(gbi_salvo_count))
        normalized_gbi_salvo_interval = max(0.0, gbi_salvo_interval)
        normalized_thaad_salvo_count = max(1, int(thaad_salvo_count))
        normalized_thaad_salvo_interval = max(0.0, thaad_salvo_interval)

        default_interceptors = [
            InterceptorConfig(
                name="GBI",
                site=interceptor_site,
                launch_delay=gbi_delay,
                engage_altitude_min=120_000.0,
                engage_altitude_max=1_200_000.0,
                engage_range_min=0.0,
                engage_range_max=6_000_000.0,
                speed_cap=gbi_speed,
                guidance_gain=gbi_gain,
                damping_gain=gbi_damping,
                intercept_distance=gbi_dist,
                max_accel=gbi_accel,
                guidance_noise_std_deg=gbi_noise,
                confusion_probability=max(0.0, min(0.15, decoy_confusion_probability * 0.5)),
                reacquisition_rate=gbi_reacq,
                max_flight_time=1200.0,
                depends_on=None,
                dependency_grace_period=0.0,
                salvo_count=normalized_gbi_salvo_count,
                salvo_interval=normalized_gbi_salvo_interval,
            ),
            InterceptorConfig(
                name="THAAD",
                site=interceptor_site,
                launch_delay=thaad_delay,
                engage_altitude_min=20_000.0,
                engage_altitude_max=220_000.0,
                engage_range_min=0.0,
                engage_range_max=800_000.0,
                speed_cap=thaad_speed,
                guidance_gain=thaad_gain,
                damping_gain=thaad_damping,
                intercept_distance=thaad_dist,
                max_accel=thaad_accel,
                guidance_noise_std_deg=thaad_noise,
                confusion_probability=min(0.12, decoy_confusion_probability + 0.03),
                reacquisition_rate=thaad_reacq,
                max_flight_time=800.0,
                depends_on="GBI",
                dependency_grace_period=45.0,
                salvo_count=normalized_thaad_salvo_count,
                salvo_interval=normalized_thaad_salvo_interval,
            ),
        ]

        for cfg in default_interceptors:
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

    air_density = air_density_factory(
        atmospheric_density_sea_level=atmospheric_density_sea_level,
        atmospheric_scale_height=atmospheric_scale_height,
        use_standard_atmosphere=use_standard_atmosphere,
    )

    # Initialize radar for discrimination
    radar_state: Optional[RadarState] = None
    if use_discrimination:
        if radar_config is not None:
            radar_state = RadarState(config=radar_config)
        else:
            # Create default radar at the interceptor site
            default_radar_config = RadarConfig(
                name="UEWR",  # Upgraded Early Warning Radar
                position=interceptor_site,
                max_range=4000_000.0,  # 4000 km
                min_rcs_at_max_range=0.001,
                update_rate=10.0,
                position_noise_std=50.0,
                velocity_noise_std=5.0,
                track_initiation_threshold=3,
                antenna_height=30.0,
            )
            radar_state = RadarState(config=default_radar_config)

    samples: List[TrajectorySample] = []
    decoy_intercepts: List[Tuple[float, Vector, str, str]] = []

    time = 0.0
    step_count = 0
    max_steps = math.inf
    if adaptive_time_step:
        min_dt = adaptive_dt_min if adaptive_dt_min is not None else max(0.02, dt * 0.2)
    else:
        min_dt = dt

    if max_time is not None:
        max_steps = int(math.ceil(max_time / min_dt))
    else:
        target_seconds = 50_000.0
        max_steps = int(math.ceil(target_seconds / min_dt))

    def _adaptive_dt_multi() -> float:
        if not adaptive_time_step:
            return dt

        max_dt = adaptive_dt_max if adaptive_dt_max is not None else dt
        min_dt = adaptive_dt_min if adaptive_dt_min is not None else max(0.02, dt * 0.2)
        step_dt = max_dt

        for state in icbm_states.values():
            if not state.launched or state.destroyed or state.impacted:
                continue
            relative_time = time - state.config.launch_time
            if state.stage_count and relative_time < state.cumulative_stage_times[-1]:
                step_dt = min(step_dt, max(min_dt, dt * 0.25))
            if state.position[1] <= 120_000.0 and state.velocity[1] < 0.0:
                step_dt = min(step_dt, max(min_dt, dt * 0.5))

        if any(state.launched and not state.expended for state in interceptor_states):
            step_dt = min(step_dt, max(min_dt, dt * 0.5))

        return max(min_dt, min(step_dt, max_dt))

    # Main simulation loop
    while True:
        step_dt = _adaptive_dt_multi()
        # Advance all ICBM states
        for name, icbm_state in icbm_states.items():
            _advance_icbm_state_multi(
                time=time,
                dt=step_dt,
                gravity=gravity,
                earth_radius=earth_radius,
                wind_velocity=wind_velocity,
                air_density=air_density,
                icbm_state=icbm_state,
                interceptor_states=interceptor_states,
                rng=rng,
            )

        new_icbms: List[Tuple[str, ICBMState]] = []
        existing_names = set(icbm_states.keys())
        for name, icbm_state in list(icbm_states.items()):
            spawned = _spawn_mirv_warheads(
                time=time, icbm_state=icbm_state, rng=rng, existing_names=existing_names
            )
            new_icbms.extend(spawned)
        if new_icbms:
            icbm_states.update(new_icbms)

        # Update radar tracking and discrimination
        if radar_state is not None and use_discrimination:
            # Update radar tracking for all visible objects
            update_radar_tracking(
                radar_state=radar_state,
                icbm_states=icbm_states,
                time=time,
                dt=step_dt,
                earth_radius=earth_radius,
                rng=rng,
            )
            
            # Update discrimination estimates (ballistic coefficient and P(warhead))
            update_discrimination(
                radar_state=radar_state,
                icbm_states=icbm_states,
                air_density_func=air_density,
                gravity=gravity,
                earth_radius=earth_radius,
            )

        # Update interceptor states
        _update_interceptor_states_multi(
            time=time,
            dt=step_dt,
            icbm_states=icbm_states,
            interceptor_states=interceptor_states,
            decoy_intercepts=decoy_intercepts,
            rng=rng,
            launcher_states=launcher_states,
            radar_state=radar_state,
            use_discrimination=use_discrimination,
        )

        # Ground clamp ICBMs
        for name, icbm_state in icbm_states.items():
            if icbm_state.position[1] < 0.0:
                icbm_state.position = (icbm_state.position[0], 0.0)

            # Ground clamp decoys
            for idx, pos in enumerate(icbm_state.decoy_state.positions):
                if pos[1] < 0.0:
                    icbm_state.decoy_state.positions[idx] = (pos[0], 0.0)
                    icbm_state.decoy_state.velocities[idx] = (icbm_state.decoy_state.velocities[idx][0], 0.0)

        # Collect samples
        _collect_samples_multi(
            time=time,
            icbm_states=icbm_states,
            interceptor_states=interceptor_states,
            samples=samples,
        )

        # Check termination conditions
        all_destroyed = all(state.destroyed for state in icbm_states.values())
        all_resolved = True
        for name, icbm_state in icbm_states.items():
            if not icbm_state.destroyed:
                if icbm_state.launched and icbm_state.position[1] <= 0.0 and time > 0.0:
                    if not icbm_state.impacted:
                        icbm_state.impacted = True
                        icbm_state.impact_time = time
                elif icbm_state.launched and not icbm_state.impacted:
                    all_resolved = False
                elif not icbm_state.launched:
                    all_resolved = False

        if all_destroyed or all_resolved:
            break

        time += step_dt
        step_count += 1
        if step_count >= max_steps:
            if max_time is not None:
                break
            raise RuntimeError(
                "Simulation exceeded safety iteration limit without intercept or ground impact. "
                "Check parameters for runaway conditions or supply max_time."
            )

    # Build outcomes
    icbm_outcomes: Dict[str, ICBMOutcome] = {}
    partial_success_count = 0
    first_intercept_time: Optional[float] = None
    first_intercept_position: Optional[Vector] = None
    first_impact_time: Optional[float] = None

    for name, icbm_state in icbm_states.items():
        destroyed = icbm_state.destroyed
        impacted = icbm_state.impacted
        escaped = not destroyed and not impacted

        intercept_time_for_icbm: Optional[float] = None
        intercept_position_for_icbm: Optional[Vector] = None

        if destroyed:
            partial_success_count += 1
            # Find the interceptor that killed this ICBM
            for state in interceptor_states:
                if state.success and state.target_icbm_name == name and state.intercept_target_label == "primary":
                    intercept_time_for_icbm = state.intercept_time
                    intercept_position_for_icbm = state.intercept_position
                    if first_intercept_time is None:
                        first_intercept_time = intercept_time_for_icbm
                        first_intercept_position = intercept_position_for_icbm
                    break

        if impacted and first_impact_time is None:
            first_impact_time = icbm_state.impact_time

        icbm_outcomes[name] = ICBMOutcome(
            name=name,
            destroyed=destroyed,
            impacted=impacted,
            escaped=escaped,
            destroyed_by=icbm_state.destroyed_by,
            impact_time=icbm_state.impact_time,
            intercept_time=intercept_time_for_icbm,
            intercept_position=intercept_position_for_icbm,
            decoys_deployed=icbm_state.decoy_state.released_count,
        )

    total_icbm_count = len(icbm_states)
    overall_success = partial_success_count == total_icbm_count

    # Build interceptor reports
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
            target_icbm_name=state.target_icbm_name,
            site_name=state.site_name,
            battery_name=state.battery_name,
        )
        for state in interceptor_states
    }

    # Build parameter record
    parameter_record: Dict[str, Any] = {
        "dt": dt,
        "max_time": max_time,
        "gravity": gravity,
        "earth_radius": earth_radius,
        "atmospheric_density_sea_level": atmospheric_density_sea_level,
        "atmospheric_scale_height": atmospheric_scale_height,
        "use_standard_atmosphere": use_standard_atmosphere,
        "adaptive_time_step": adaptive_time_step,
        "adaptive_dt_min": adaptive_dt_min,
        "adaptive_dt_max": adaptive_dt_max,
        "wind_velocity": wind_velocity,
        "icbm_count": len(icbm_configs),
        "icbm_total_count": total_icbm_count,
        "icbm_configs": [asdict(cfg) for cfg in icbm_configs],
        "defense_sites": [asdict(site) for site in defense_sites] if defense_sites else None,
    }

    # Total decoys deployed
    total_decoy_count = sum(state.decoy_state.released_count for state in icbm_states.values())

    # Legacy decoy_intercepts format (time, pos, interceptor_label)
    legacy_decoy_intercepts = [(t, pos, label) for t, pos, label, _ in decoy_intercepts]

    return SimulationResult(
        intercept_success=overall_success if len(icbm_configs) == 1 else (partial_success_count > 0),
        intercept_time=first_intercept_time,
        intercept_position=first_intercept_position,
        icbm_impact_time=first_impact_time,
        samples=samples,
        intercept_target_label="primary" if partial_success_count > 0 else None,
        decoy_intercepts=legacy_decoy_intercepts,
        decoy_count=total_decoy_count,
        parameters=parameter_record,
        interceptor_reports=interceptor_reports,
        icbm_outcomes=icbm_outcomes,
        overall_success=overall_success,
        partial_success_count=partial_success_count,
        total_icbm_count=total_icbm_count,
        defense_sites=list(defense_sites) if defense_sites else [],
        radar_tracks_count=len(radar_state.tracked_objects) if radar_state else 0,
        max_discrimination_confidence=max(
            [obj.warhead_probability for obj in radar_state.tracked_objects.values()] + [0.0]
        ) if radar_state else 0.0,
    )


def simulate_icbm_intercept(
    *,
    dt: float = 0.25,
    max_time: Optional[float] = None,
    gravity: float = 9.81,
    earth_radius: float = 6_371_000.0,
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
    use_standard_atmosphere: bool = True,
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
    icbm_rcs: Optional[float] = None,
    warhead_ballistic_coeff: Optional[float] = None,
    mirv_count: int = 1,
    mirv_release_time: Optional[float] = None,
    mirv_spread_velocity: float = 120.0,
    adaptive_time_step: bool = False,
    adaptive_dt_min: Optional[float] = None,
    adaptive_dt_max: Optional[float] = None,
    rng: Optional[Random] = None,
    interceptors: Optional[List[InterceptorConfig]] = None,
    # Multi-ICBM support
    icbm_configs: Optional[List[ICBMConfig]] = None,
    # Multi-site defense support
    defense_sites: Optional[List[DefenseSiteConfig]] = None,
    # Radar and discrimination
    use_discrimination: bool = True,
    radar_config: Optional[RadarConfig] = None,
    # Layer-specific interceptor overrides
    gbi_speed_cap: Optional[float] = None,
    thaad_speed_cap: Optional[float] = None,
    gbi_launch_delay: Optional[float] = None,
    thaad_launch_delay: Optional[float] = None,
    gbi_guidance_gain: Optional[float] = None,
    thaad_guidance_gain: Optional[float] = None,
    gbi_damping_gain: Optional[float] = None,
    thaad_damping_gain: Optional[float] = None,
    gbi_intercept_distance: Optional[float] = None,
    thaad_intercept_distance: Optional[float] = None,
    gbi_max_accel: Optional[float] = None,
    thaad_max_accel: Optional[float] = None,
    gbi_guidance_noise: Optional[float] = None,
    thaad_guidance_noise: Optional[float] = None,
    gbi_reacquisition_rate: Optional[float] = None,
    thaad_reacquisition_rate: Optional[float] = None,
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
    * Optional MIRV release that spawns multiple warheads in the multi-ICBM model.
    * Optional adaptive time stepping for boost/reentry precision.
    * US Standard Atmosphere 1976 density by default (configurable).
    """
    if dt <= 0.0:
        raise ValueError("time step dt must be positive")

    if rng is None:
        rng = random.Random()

    # Apply any ICBM/MIRV overrides if icbm_configs is provided
    if icbm_configs is not None:
        icbm_overrides = {}
        # Only override if values are explicitly provided (not None or default in some cases)
        # Note: for mirv_count, we check if it's > 1 as that's the current logic
        if mirv_count > 1:
            icbm_overrides["mirv_count"] = mirv_count
        if mirv_release_time is not None:
            icbm_overrides["mirv_release_time"] = mirv_release_time
        if mirv_spread_velocity != 120.0: # Only if changed from default
            icbm_overrides["mirv_spread_velocity"] = mirv_spread_velocity
        
        # New overrides
        if icbm_rcs is not None:
            icbm_overrides["rcs"] = icbm_rcs
        if warhead_ballistic_coeff is not None:
            icbm_overrides["warhead_ballistic_coeff"] = warhead_ballistic_coeff
            
        # Also apply decoy overrides if provided (these have defaults in the signature, 
        # so we need to be careful. But simulate_icbm_intercept is called with these 
        # defaults if not provided in kwargs.)
        # However, for Monte Carlo, we want these to be applied.
        
        # To match main()'s behavior, we only apply if they differ from "standard" defaults 
        # or if we want to force them.
        
        # Let's just apply them all if icbm_configs is present, to ensure consistency.
        # But wait, create_mixed_salvo already set some of these.
        
        # The cleanest way is to only apply if they were explicitly passed to simulate_icbm_intercept.
        # But in Python we can't easily tell if it was default or passed unless we use a sentinel.
        
        # For now, let's only apply the ones that were added as Optional[float] = None.
        
        if icbm_overrides:
            adjusted_configs: List[ICBMConfig] = []
            for cfg in icbm_configs:
                cfg_data = dict(cfg.__dict__)
                cfg_data.update(icbm_overrides)
                adjusted_configs.append(ICBMConfig(**cfg_data))
            icbm_configs = adjusted_configs

    if icbm_configs is not None or defense_sites is not None or mirv_count > 1:
        return _simulate_multi_icbm(
            dt=dt,
            max_time=max_time,
            gravity=gravity,
            earth_radius=earth_radius,
            atmospheric_density_sea_level=atmospheric_density_sea_level,
            atmospheric_scale_height=atmospheric_scale_height,
            use_standard_atmosphere=use_standard_atmosphere,
            adaptive_time_step=adaptive_time_step,
            adaptive_dt_min=adaptive_dt_min,
            adaptive_dt_max=adaptive_dt_max,
            wind_velocity=wind_velocity,
            rng=rng,
            icbm_configs=icbm_configs or [
                ICBMConfig(
                    name="ICBM-1",
                    start_position=icbm_start,
                    initial_velocity=icbm_velocity,
                    mass=icbm_mass,
                    drag_coefficient=icbm_drag_coefficient,
                    reference_area=icbm_reference_area,
                    boost_profile=icbm_boost_profile,
                    pitch_schedule_deg=icbm_pitch_schedule_deg,
                    decoy_count=decoy_count,
                    decoy_release_time=decoy_release_time,
                    decoy_spread_velocity=decoy_spread_velocity,
                    decoy_drag_multiplier=decoy_drag_multiplier,
                    decoy_mass_fraction=decoy_mass_fraction,
                    warhead_mass_fraction=warhead_mass_fraction,
                    warhead_drag_multiplier=warhead_drag_multiplier,
                    mirv_count=mirv_count,
                    mirv_release_time=mirv_release_time,
                    mirv_spread_velocity=mirv_spread_velocity,
                    rcs=icbm_rcs if icbm_rcs is not None else 0.05,
                    warhead_ballistic_coeff=warhead_ballistic_coeff,
                )
            ],
            defense_sites=defense_sites,
            interceptors=interceptors,
            interceptor_site=interceptor_site,
            interceptor_speed_cap=interceptor_speed_cap,
            interceptor_launch_delay=interceptor_launch_delay,
            gbi_salvo_count=gbi_salvo_count,
            gbi_salvo_interval=gbi_salvo_interval,
            thaad_salvo_count=thaad_salvo_count,
            thaad_salvo_interval=thaad_salvo_interval,
            guidance_gain=guidance_gain,
            damping_gain=damping_gain,
            intercept_distance=intercept_distance,
            guidance_noise_std_deg=guidance_noise_std_deg,
            interceptor_max_accel=interceptor_max_accel,
            decoy_confusion_probability=decoy_confusion_probability,
            decoy_reacquisition_rate=decoy_reacquisition_rate,
            icbm_rcs_override=icbm_rcs,
            warhead_ballistic_coeff_override=warhead_ballistic_coeff,
            use_discrimination=use_discrimination,
            radar_config=radar_config,
            gbi_speed_cap_override=gbi_speed_cap,
            thaad_speed_cap_override=thaad_speed_cap,
            gbi_launch_delay_override=gbi_launch_delay,
            thaad_launch_delay_override=thaad_launch_delay,
            gbi_guidance_gain_override=gbi_guidance_gain,
            thaad_guidance_gain_override=thaad_guidance_gain,
            gbi_damping_gain_override=gbi_damping_gain,
            thaad_damping_gain_override=thaad_damping_gain,
            gbi_intercept_distance_override=gbi_intercept_distance,
            thaad_intercept_distance_override=thaad_intercept_distance,
            gbi_max_accel_override=gbi_max_accel,
            thaad_max_accel_override=thaad_max_accel,
            gbi_guidance_noise_override=gbi_guidance_noise,
            thaad_guidance_noise_override=thaad_guidance_noise,
            gbi_reacquisition_rate_override=gbi_reacquisition_rate,
            thaad_reacquisition_rate_override=thaad_reacquisition_rate,
        )

    # ========================================================================
    # Legacy single-ICBM mode
    # ========================================================================
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
        # Use provided layer-specific overrides or fall back to defaults/generic params
        gbi_speed = gbi_speed_cap if gbi_speed_cap is not None else interceptor_speed_cap
        gbi_delay = gbi_launch_delay if gbi_launch_delay is not None else interceptor_launch_delay
        gbi_gain = gbi_guidance_gain if gbi_guidance_gain is not None else max(0.88, guidance_gain * 1.18)
        gbi_damping = gbi_damping_gain if gbi_damping_gain is not None else max(0.055, damping_gain * 1.15)
        gbi_dist = gbi_intercept_distance if gbi_intercept_distance is not None else max(intercept_distance, 96_000.0)
        gbi_accel = gbi_max_accel if gbi_max_accel is not None else interceptor_max_accel
        gbi_noise = gbi_guidance_noise if gbi_guidance_noise is not None else max(0.03, guidance_noise_std_deg * 0.9)
        gbi_reacq = gbi_reacquisition_rate if gbi_reacquisition_rate is not None else max(decoy_reacquisition_rate, 0.018)

        thaad_speed = thaad_speed_cap if thaad_speed_cap is not None else 5000.0
        thaad_delay = thaad_launch_delay if thaad_launch_delay is not None else interceptor_launch_delay + 220.0
        thaad_gain = thaad_guidance_gain if thaad_guidance_gain is not None else max(0.68, guidance_gain * 1.38)
        thaad_damping = thaad_damping_gain if thaad_damping_gain is not None else max(0.11, damping_gain * 1.9)
        thaad_dist = thaad_intercept_distance if thaad_intercept_distance is not None else 180_000.0
        thaad_accel = thaad_max_accel if thaad_max_accel is not None else max(interceptor_max_accel, 155.0)
        thaad_noise = thaad_guidance_noise if thaad_guidance_noise is not None else max(0.035, guidance_noise_std_deg * 1.05)
        thaad_reacq = thaad_reacquisition_rate if thaad_reacquisition_rate is not None else max(decoy_reacquisition_rate * 2.0, 0.06)

        interceptors = [
            InterceptorConfig(
                name="GBI",
                site=interceptor_site,
                launch_delay=gbi_delay,
                engage_altitude_min=120_000.0,
                engage_altitude_max=1_200_000.0,
                engage_range_min=0.0,
                engage_range_max=6_000_000.0,
                speed_cap=gbi_speed,
                guidance_gain=gbi_gain,
                damping_gain=gbi_damping,
                intercept_distance=gbi_dist,
                max_accel=gbi_accel,
                guidance_noise_std_deg=gbi_noise,
                confusion_probability=max(0.0, min(0.15, decoy_confusion_probability * 0.5)),
                reacquisition_rate=gbi_reacq,
                max_flight_time=1200.0,
                depends_on=None,
                dependency_grace_period=0.0,
                salvo_count=normalized_gbi_salvo_count,
                salvo_interval=normalized_gbi_salvo_interval,
            ),
            InterceptorConfig(
                name="THAAD",
                site=interceptor_site,
                launch_delay=thaad_delay,
                engage_altitude_min=20_000.0,
                engage_altitude_max=220_000.0,
                engage_range_min=0.0,
                engage_range_max=800_000.0,
                speed_cap=thaad_speed,
                guidance_gain=thaad_gain,
                damping_gain=thaad_damping,
                intercept_distance=thaad_dist,
                max_accel=thaad_accel,
                guidance_noise_std_deg=thaad_noise,
                confusion_probability=min(0.12, decoy_confusion_probability + 0.03),
                reacquisition_rate=thaad_reacq,
                max_flight_time=800.0,
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
        "earth_radius": earth_radius,
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
        "use_standard_atmosphere": use_standard_atmosphere,
        "adaptive_time_step": adaptive_time_step,
        "adaptive_dt_min": adaptive_dt_min,
        "adaptive_dt_max": adaptive_dt_max,
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
        "mirv_count": mirv_count,
        "mirv_release_time": mirv_release_time,
        "mirv_spread_velocity": mirv_spread_velocity,
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

    air_density = air_density_factory(
        atmospheric_density_sea_level=atmospheric_density_sea_level,
        atmospheric_scale_height=atmospheric_scale_height,
        use_standard_atmosphere=use_standard_atmosphere,
    )

    # Internal state copies so we do not mutate caller defaults.
    icbm_pos = icbm_start
    icbm_vel = icbm_velocity

    samples: List[TrajectorySample] = []

    intercept_success = False
    intercept_time: Optional[float] = None
    intercept_position: Optional[Vector] = None
    icbm_impact_time: Optional[float] = None
    intercept_target_label: Optional[str] = None
    decoy_intercepts: List[Tuple[float, Vector, str]] = []

    active_drag_coefficient = icbm_drag_coefficient
    active_reference_area = icbm_reference_area
    active_mass = icbm_mass

    decoy_state = DecoyState(
        positions=[],
        velocities=[],
        masses=[],
        drag_coefficients=[],
        reference_areas=[],
        ids=[],
    )

    time = 0.0
    step_count = 0
    max_steps = math.inf
    if adaptive_time_step:
        min_dt = adaptive_dt_min if adaptive_dt_min is not None else max(0.02, dt * 0.2)
    else:
        min_dt = dt

    if max_time is not None:
        max_steps = int(math.ceil(max_time / min_dt))
    else:
        # Safety guard: maintain roughly 50,000 simulated seconds regardless of dt.
        target_seconds = 50_000.0
        max_steps = int(math.ceil(target_seconds / min_dt))

    def _adaptive_dt_single() -> float:
        if not adaptive_time_step:
            return dt

        max_dt = adaptive_dt_max if adaptive_dt_max is not None else dt
        min_local = adaptive_dt_min if adaptive_dt_min is not None else max(0.02, dt * 0.2)
        step_dt = max_dt

        if stage_count and time < cumulative_stage_times[-1]:
            step_dt = min(step_dt, max(min_local, dt * 0.25))
        if icbm_pos[1] <= 120_000.0 and icbm_vel[1] < 0.0:
            step_dt = min(step_dt, max(min_local, dt * 0.5))
        if any(state.launched and not state.expended for state in interceptor_states):
            step_dt = min(step_dt, max(min_local, dt * 0.5))

        return max(min_local, min(step_dt, max_dt))

    while True:
        step_dt = _adaptive_dt_single()
        (
            icbm_pos,
            icbm_vel,
            active_mass,
            active_drag_coefficient,
            active_reference_area,
            decoy_state,
        ) = advance_icbm_state(
            time=time,
            dt=step_dt,
            gravity=gravity,
            earth_radius=earth_radius,
            wind_velocity=wind_velocity,
            air_density=air_density,
            icbm_pos=icbm_pos,
            icbm_vel=icbm_vel,
            initial_heading=initial_heading,
            stage_count=stage_count,
            cumulative_stage_times=cumulative_stage_times,
            stage_accels=stage_accels,
            cumulative_pitch_angles=cumulative_pitch_angles,
            active_mass=active_mass,
            active_drag_coefficient=active_drag_coefficient,
            active_reference_area=active_reference_area,
            icbm_mass=icbm_mass,
            icbm_drag_coefficient=icbm_drag_coefficient,
            icbm_reference_area=icbm_reference_area,
            decoy_release_time=decoy_release_time,
            decoy_count=decoy_count,
            decoy_spread_velocity=decoy_spread_velocity,
            decoy_drag_multiplier=decoy_drag_multiplier,
            warhead_mass_fraction=warhead_mass_fraction,
            warhead_drag_multiplier=warhead_drag_multiplier,
            decoy_mass_fraction=decoy_mass_fraction,
            decoy_state=decoy_state,
            interceptor_states=interceptor_states,
            rng=rng,
        )

        (
            intercept_success,
            intercept_time,
            intercept_position,
            intercept_target_label,
            decoy_state,
        ) = update_interceptor_states(
            time=time,
            dt=step_dt,
            icbm_pos=icbm_pos,
            icbm_vel=icbm_vel,
            interceptor_states=interceptor_states,
            intercept_success=intercept_success,
            intercept_time=intercept_time,
            intercept_position=intercept_position,
            intercept_target_label=intercept_target_label,
            decoy_intercepts=decoy_intercepts,
            decoy_state=decoy_state,
            rng=rng,
        )

        if icbm_pos[1] < 0.0:
            icbm_pos = (icbm_pos[0], 0.0)

        for idx, pos in enumerate(decoy_state.positions):
            if pos[1] < 0.0:
                decoy_state.positions[idx] = (pos[0], 0.0)
                decoy_state.velocities[idx] = (decoy_state.velocities[idx][0], 0.0)

        collect_samples(
            time=time,
            icbm_pos=icbm_pos,
            icbm_vel=icbm_vel,
            interceptor_states=interceptor_states,
            interceptors=interceptors,
            decoy_state=decoy_state,
            samples=samples,
        )

        if intercept_success:
            break

        if icbm_pos[1] <= 0.0 and time > 0.0:
            icbm_impact_time = time
            break

        time += step_dt
        step_count += 1
        if step_count >= max_steps:
            if max_time is not None:
                break
            raise RuntimeError(
                "Simulation exceeded safety iteration limit without intercept or ground impact. "
                "Check parameters for runaway conditions or supply max_time."
            )

    parameter_record["decoys_deployed"] = decoy_state.released_count
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
            target_icbm_name=state.target_icbm_name,
            site_name=state.site_name,
            battery_name=state.battery_name,
        )
        for state in interceptor_states
    }

    # Build legacy single-ICBM outcome
    single_icbm_outcome = ICBMOutcome(
        name="ICBM-1",
        destroyed=intercept_success,
        impacted=icbm_impact_time is not None,
        escaped=not intercept_success and icbm_impact_time is None,
        destroyed_by=next((s.label for s in interceptor_states if s.success and s.intercept_target_label == "primary"), None),
        impact_time=icbm_impact_time,
        intercept_time=intercept_time,
        intercept_position=intercept_position,
        decoys_deployed=decoy_state.released_count,
    )

    return SimulationResult(
        intercept_success=intercept_success,
        intercept_time=intercept_time,
        intercept_position=intercept_position,
        icbm_impact_time=icbm_impact_time,
        samples=samples,
        intercept_target_label=intercept_target_label,
        decoy_intercepts=decoy_intercepts,
        decoy_count=decoy_state.released_count,
        parameters=parameter_record,
        interceptor_reports=interceptor_reports,
        icbm_outcomes={"ICBM-1": single_icbm_outcome},
        overall_success=intercept_success,
        partial_success_count=1 if intercept_success else 0,
        total_icbm_count=1,
        defense_sites=[],
        radar_tracks_count=0,  # No radar in legacy mode
        max_discrimination_confidence=0.0,
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
        "overall_success": result.overall_success,
        "partial_success_count": result.partial_success_count,
        "total_icbm_count": result.total_icbm_count,
        "target": result.intercept_target_label,
        "intercept_time": result.intercept_time,
        "impact_time": result.icbm_impact_time,
        "decoy_count": result.decoy_count,
        "radar_tracks_count": result.radar_tracks_count,
        "max_discrimination_confidence": result.max_discrimination_confidence,
        "decoy_intercepts": [
            {
                "time": time,
                "position": list(position),
                "interceptor": interceptor_name,
            }
            for time, position, interceptor_name in result.decoy_intercepts
        ],
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
    decoy_events = sorted(result.decoy_intercepts, key=lambda entry: entry[0])
    decoy_note = None
    if decoy_events:
        decoy_time, decoy_position, decoy_interceptor = decoy_events[0]
        dx, dy = decoy_position
        decoy_prefix = f"{decoy_interceptor} interceptor" if decoy_interceptor else "Interceptor"
        decoy_note = (
            f"{decoy_prefix} collided with a decoy at t={decoy_time:6.1f}s "
            f"over position ({dx:,.0f} m, {dy:,.0f} m)"
        )
    if result.intercept_success and result.intercept_time is not None:
        x, y = result.intercept_position or (0.0, 0.0)
        interceptor_name = None
        for name, report in result.interceptor_reports.items():
            if report.success and report.target_label == "primary":
                interceptor_name = name
                break
        prefix = f"{interceptor_name} interceptor" if interceptor_name else "Interceptor"
        primary_message = (
            f"{prefix} achieved lock at t={result.intercept_time:6.1f}s "
            f"over position ({x:,.0f} m, {y:,.0f} m)."
        )
        if decoy_note:
            return f"{primary_message} Earlier, {decoy_note}."
        return primary_message

    if result.icbm_impact_time is not None:
        if decoy_note:
            return (
                f"{decoy_note}; primary warhead impacted at t={result.icbm_impact_time:6.1f}s."
            )
        return (
            "Interceptor failed to engage before impact. "
            f"ICBM reached ground at t={result.icbm_impact_time:6.1f}s."
        )

    if decoy_note:
        return f"{decoy_note}; primary warhead continued."

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
                desc += ", expended without intercept (timeout)"
        else:
            if icbm_impact_time is not None:
                desc += f", still in flight when impact occurred at t={icbm_impact_time:6.1f}s"
            else:
                desc += ", still active at simulation end"
    return desc


def _engagement_statistics(result: SimulationResult) -> str:
    """Compute and format engagement statistics from interceptor reports."""
    reports = result.interceptor_reports
    
    # Count by layer
    layer_stats: Dict[str, Dict[str, int]] = {}
    
    for name, report in reports.items():
        layer = report.config_name
        if layer not in layer_stats:
            layer_stats[layer] = {
                "total": 0,
                "launched": 0,
                "primary_kill": 0,
                "decoy_hit": 0,
                "timeout": 0,
                "never_launched": 0,
            }
        
        layer_stats[layer]["total"] += 1
        
        if report.launch_time is None:
            layer_stats[layer]["never_launched"] += 1
        else:
            layer_stats[layer]["launched"] += 1
            
            if report.success and report.target_label == "primary":
                layer_stats[layer]["primary_kill"] += 1
            elif report.target_label == "decoy":
                layer_stats[layer]["decoy_hit"] += 1
            elif report.expended and not report.success:
                layer_stats[layer]["timeout"] += 1
    
    # Format the output
    lines = ["Engagement Statistics:"]
    for layer, stats in sorted(layer_stats.items()):
        launched = stats["launched"]
        total = stats["total"]
        primary = stats["primary_kill"]
        decoy = stats["decoy_hit"]
        timeout = stats["timeout"]
        never = stats["never_launched"]
        
        line = f"  {layer}: {launched}/{total} launched"
        outcomes = []
        if primary > 0:
            outcomes.append(f"{primary} primary kill")
        if decoy > 0:
            outcomes.append(f"{decoy} decoy hit")
        if timeout > 0:
            outcomes.append(f"{timeout} timeout")
        if never > 0:
            outcomes.append(f"{never} not launched")
        
        if outcomes:
            line += f" ({', '.join(outcomes)})"
        lines.append(line)
    
    return "\n".join(lines)


def _monte_carlo_bases(base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "launch_delay_base": base_kwargs.get("interceptor_launch_delay", 120.0),
        "speed_cap_base": base_kwargs.get("interceptor_speed_cap", 5000.0),
        "noise_base": base_kwargs.get("guidance_noise_std_deg", 0.10),
        "wind_base": base_kwargs.get("wind_velocity", (0.0, 0.0)),
        "gbi_salvo_base": max(1, int(base_kwargs.get("gbi_salvo_count", 1))),
        "gbi_salvo_interval_base": max(0.0, base_kwargs.get("gbi_salvo_interval", 0.0)),
        "thaad_salvo_base": max(1, int(base_kwargs.get("thaad_salvo_count", 1))),
        "thaad_salvo_interval_base": max(0.0, base_kwargs.get("thaad_salvo_interval", 0.0)),
        "boost_profile_base": tuple(base_kwargs.get("icbm_boost_profile", DEFAULT_BOOST_PROFILE)),
        "pitch_base": tuple(base_kwargs.get("icbm_pitch_schedule_deg", DEFAULT_PITCH_SCHEDULE_DEG)),
        "accel_base": base_kwargs.get("interceptor_max_accel", 60.0),
        "decoy_release_base": base_kwargs.get("decoy_release_time", 220.0),
        "decoy_count_base": int(base_kwargs.get("decoy_count", 3)),
        "decoy_spread_base": base_kwargs.get("decoy_spread_velocity", 280.0),
        "decoy_confusion_base": base_kwargs.get("decoy_confusion_probability", 0.1),
        "decoy_reacquire_base": base_kwargs.get("decoy_reacquisition_rate", 0.015),
        "warhead_mass_base": base_kwargs.get("warhead_mass_fraction", 0.35),
        "warhead_drag_base": base_kwargs.get("warhead_drag_multiplier", 0.6),
        "decoy_mass_base": base_kwargs.get("decoy_mass_fraction", 0.04),
        "decoy_drag_base": base_kwargs.get("decoy_drag_multiplier", 4.0),
    }


def _monte_carlo_worker(
    run_index: int,
    run_seed: int,
    base_kwargs: Dict[str, Any],
    bases: Dict[str, Any],
    include_details: bool,
) -> Dict[str, Any]:
    run_rng = random.Random(run_seed)
    kwargs = dict(base_kwargs)
    kwargs["rng"] = run_rng
    kwargs["interceptor_launch_delay"] = max(
        10.0, run_rng.gauss(bases["launch_delay_base"], max(5.0, 0.2 * abs(bases["launch_delay_base"])))
    )
    kwargs["interceptor_speed_cap"] = max(
        1500.0, run_rng.gauss(bases["speed_cap_base"], max(200.0, 0.15 * abs(bases["speed_cap_base"])))
    )
    kwargs["guidance_noise_std_deg"] = max(
        0.0, run_rng.gauss(bases["noise_base"], max(0.05, 0.3 * bases["noise_base"] if bases["noise_base"] else 0.1))
    )
    kwargs["gbi_salvo_count"] = bases["gbi_salvo_base"]
    kwargs["gbi_salvo_interval"] = bases["gbi_salvo_interval_base"]
    kwargs["thaad_salvo_count"] = bases["thaad_salvo_base"]
    kwargs["thaad_salvo_interval"] = bases["thaad_salvo_interval_base"]
    wind_base = bases["wind_base"]
    kwargs["wind_velocity"] = (
        run_rng.gauss(wind_base[0], 80.0),
        run_rng.gauss(wind_base[1], 12.0),
    )
    profile_variation: List[Tuple[float, float]] = []
    for idx, (duration, accel) in enumerate(bases["boost_profile_base"]):
        dur_sigma = max(5.0, 0.15 * duration)
        acc_sigma = max(0.5, 0.25 * max(abs(accel), 1.0))
        duration_sample = max(5.0, run_rng.gauss(duration, dur_sigma))
        accel_sample = run_rng.gauss(accel, acc_sigma)
        profile_variation.append((duration_sample, accel_sample))
    kwargs["icbm_boost_profile"] = tuple(profile_variation)

    pitch_base = bases["pitch_base"]
    if pitch_base:
        pitch_samples: List[float] = []
        last_angle = pitch_base[-1]
        for idx in range(len(profile_variation)):
            base_angle = pitch_base[idx] if idx < len(pitch_base) else last_angle
            pitch_samples.append(run_rng.gauss(base_angle, 1.5))
        kwargs["icbm_pitch_schedule_deg"] = tuple(pitch_samples)

    kwargs["interceptor_max_accel"] = max(
        5.0, run_rng.gauss(bases["accel_base"], max(1.0, 0.2 * max(bases["accel_base"], 1.0)))
    )
    decoy_release_base = bases["decoy_release_base"]
    if decoy_release_base is None:
        kwargs["decoy_release_time"] = None
    else:
        kwargs["decoy_release_time"] = max(20.0, run_rng.gauss(decoy_release_base, 30.0))
    decoy_count_base = bases["decoy_count_base"]
    kwargs["decoy_count"] = max(0, int(round(run_rng.gauss(decoy_count_base, max(1.0, 0.5 * decoy_count_base)))))
    decoy_spread_base = bases["decoy_spread_base"]
    kwargs["decoy_spread_velocity"] = max(
        50.0, run_rng.gauss(decoy_spread_base, 0.25 * max(decoy_spread_base, 1.0))
    )
    decoy_confusion_base = bases["decoy_confusion_base"]
    kwargs["decoy_confusion_probability"] = min(
        0.4, max(0.0, run_rng.gauss(decoy_confusion_base, 0.08))
    )
    decoy_reacquire_base = bases["decoy_reacquire_base"]
    kwargs["decoy_reacquisition_rate"] = max(
        0.0, run_rng.gauss(decoy_reacquire_base, 0.5 * max(decoy_reacquire_base, 0.005))
    )
    warhead_mass_base = bases["warhead_mass_base"]
    kwargs["warhead_mass_fraction"] = min(
        0.9, max(0.05, run_rng.gauss(warhead_mass_base, 0.05))
    )
    warhead_drag_base = bases["warhead_drag_base"]
    kwargs["warhead_drag_multiplier"] = max(
        0.1, run_rng.gauss(warhead_drag_base, 0.1)
    )
    decoy_mass_base = bases["decoy_mass_base"]
    kwargs["decoy_mass_fraction"] = max(
        0.005, run_rng.gauss(decoy_mass_base, 0.01)
    )
    decoy_drag_base = bases["decoy_drag_base"]
    kwargs["decoy_drag_multiplier"] = max(
        0.5, run_rng.gauss(decoy_drag_base, 0.6)
    )

    result = simulate_icbm_intercept(**kwargs)

    miss_distance = _minimum_miss_distance(result)
    layer_primary_counts: Dict[str, int] = {}
    layer_decoy_counts: Dict[str, int] = {}
    decoy_intercepts = 0
    for report in result.interceptor_reports.values():
        if report.target_label == "primary" and report.success:
            base = report.config_name
            layer_primary_counts[base] = layer_primary_counts.get(base, 0) + 1
        if report.target_label == "decoy":
            base = report.config_name
            layer_decoy_counts[base] = layer_decoy_counts.get(base, 0) + 1
            decoy_intercepts += 1

    details_entry = None
    if include_details:
        record_kwargs = dict(kwargs)
        record_kwargs.pop("rng", None)
        details_entry = _result_to_entry(
            result,
            mode="monte_carlo",
            run_index=run_index,
            seed=run_seed,
            min_distance=miss_distance,
        )
        details_entry["drawn_parameters"] = _json_safe(record_kwargs)

    return {
        "run_index": run_index,
        "run_seed": run_seed,
        "intercept_success": result.intercept_success,
        "intercept_time": result.intercept_time,
        "impact_time": result.icbm_impact_time,
        "miss_distance": miss_distance,
        "decoy_intercepts": decoy_intercepts,
        "layer_primary_counts": layer_primary_counts,
        "layer_decoy_counts": layer_decoy_counts,
        "details_entry": details_entry,
    }


def run_monte_carlo(
    runs: int,
    *,
    seed: Optional[int] = None,
    base_kwargs: Optional[Dict[str, Any]] = None,
    details: Optional[List[Dict[str, Any]]] = None,
    max_workers: Optional[int] = None,
) -> MonteCarloSummary:
    if runs <= 0:
        raise ValueError("runs must be positive")

    if max_workers is None:
        max_workers = 1
    if max_workers == 0:
        max_workers = os.cpu_count() or 1
    if max_workers < 0:
        raise ValueError("max_workers must be >= 0")

    master_rng = random.Random(seed)
    base_kwargs = dict(base_kwargs or {})
    bases = _monte_carlo_bases(base_kwargs)
    include_details = details is not None

    run_seeds = [master_rng.randint(0, 2**31 - 1) for _ in range(runs)]

    if max_workers <= 1 or runs == 1:
        results = [
            _monte_carlo_worker(run_index, run_seed, base_kwargs, bases, include_details)
            for run_index, run_seed in enumerate(run_seeds)
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _monte_carlo_worker,
                    run_index,
                    run_seed,
                    base_kwargs,
                    bases,
                    include_details,
                )
                for run_index, run_seed in enumerate(run_seeds)
            ]
            results = [future.result() for future in futures]

    successes = impacts = timeouts = 0
    decoy_intercepts = 0
    intercept_times: List[float] = []
    miss_distances: List[float] = []
    layer_primary_counts: Dict[str, int] = {}
    layer_decoy_counts: Dict[str, int] = {}

    results.sort(key=lambda item: item["run_index"])
    if details is not None:
        for item in results:
            if item["details_entry"] is not None:
                details.append(item["details_entry"])

    for item in results:
        if item["intercept_success"] and item["intercept_time"] is not None:
            successes += 1
            intercept_times.append(item["intercept_time"])
        else:
            if item["impact_time"] is not None:
                impacts += 1
            else:
                timeouts += 1

        miss_distance = item["miss_distance"]
        if not item["intercept_success"] and math.isfinite(miss_distance):
            miss_distances.append(miss_distance)

        decoy_intercepts += item["decoy_intercepts"]
        for layer, count in item["layer_primary_counts"].items():
            layer_primary_counts[layer] = layer_primary_counts.get(layer, 0) + count
        for layer, count in item["layer_decoy_counts"].items():
            layer_decoy_counts[layer] = layer_decoy_counts.get(layer, 0) + count

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

    # Check if we have multi-ICBM data
    has_multi_icbm = result.samples and result.samples[0].icbm_positions

    interceptor_series, intercept_sample_indices = _interceptor_time_series(result)

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

    plt.figure(figsize=(12, 6) if has_multi_icbm else (10, 5))

    # Plot ICBM trajectories
    if has_multi_icbm:
        icbm_cmap = plt.get_cmap("Set1")
        icbm_names = sorted(result.samples[0].icbm_positions.keys())
        for icbm_idx, icbm_name in enumerate(icbm_names):
            icbm_xs: List[float] = []
            icbm_ys: List[float] = []
            for sample in result.samples:
                pos = sample.icbm_positions.get(icbm_name)
                if pos:
                    icbm_xs.append(pos[0])
                    icbm_ys.append(pos[1])
                else:
                    icbm_xs.append(math.nan)
                    icbm_ys.append(math.nan)
            color = icbm_cmap(icbm_idx % icbm_cmap.N)
            outcome = result.icbm_outcomes.get(icbm_name)
            status = ""
            if outcome:
                if outcome.destroyed:
                    status = " (destroyed)"
                elif outcome.impacted:
                    status = " (impacted)"
            plt.plot(icbm_xs, icbm_ys, color=color, linewidth=2.0, label=f"{icbm_name}{status}")
    else:
        icbm_x = [sample.icbm_position[0] for sample in result.samples]
        icbm_y = [sample.icbm_position[1] for sample in result.samples]
        plt.plot(icbm_x, icbm_y, color="#4D4D4D", linewidth=2.0, label="ICBM")

    for name, (xs, ys) in interceptor_series.items():
        intercept_idx = intercept_sample_indices.get(name)
        limit = len(xs) if intercept_idx is None else intercept_idx + 1
        x_vals: List[float] = []
        y_vals: List[float] = []
        for idx in range(limit):
            x_val = xs[idx]
            y_val = ys[idx]
            if math.isnan(x_val) or math.isnan(y_val):
                continue
            x_vals.append(x_val)
            y_vals.append(y_val)

        if not x_vals:
            continue
        report = result.interceptor_reports[name]
        style = _interceptor_style(report.config_name)
        plt.plot(x_vals, y_vals, label=f"{name}", **style)

    # Plot decoys - consolidated legend entry for cleaner display
    if decoy_id_order:
        decoy_color = "#888888"  # Gray for all decoys
        decoy_plotted = False
        for display_idx, decoy_id in enumerate(decoy_id_order, start=1):
            xs, ys = decoy_paths[decoy_id]
            if all(math.isnan(x) for x in xs):
                continue
            # Only add legend entry for the first decoy
            label = f"Decoys ({len(decoy_id_order)})" if not decoy_plotted else None
            plt.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=0.6,
                color=decoy_color,
                alpha=0.7,
                label=label,
            )
            decoy_plotted = True

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

    # Plot defense site markers
    if result.defense_sites:
        for site in result.defense_sites:
            plt.scatter(
                [site.position[0]], [0.0],
                color="green", marker="^", s=150,
                label=f"{site.name}",
                zorder=5,
            )

    plt.axhline(0.0, color="black", linewidth=0.6)
    plt.xlabel("Range (m)")
    plt.ylabel("Altitude (m)")

    # Update title based on scenario
    if has_multi_icbm:
        destroyed = result.partial_success_count
        total = result.total_icbm_count
        plt.title(f"Multi-ICBM Intercept Simulation ({destroyed}/{total} destroyed)")
    else:
        plt.title("ICBM Intercept Simulation")

    # Improved legend - use multiple columns for many items
    num_legend_items = (
        (len(result.samples[0].icbm_positions) if has_multi_icbm else 1) +
        len(interceptor_series) +
        (1 if decoy_id_order else 0) +
        len([r for r in result.interceptor_reports.values() if r.intercept_position]) +
        len(result.defense_sites)
    )
    ncol = 2 if num_legend_items > 8 else 1
    plt.legend(
        loc="upper right",
        fontsize=7,
        ncol=ncol,
        framealpha=0.9,
        borderaxespad=0.5,
    )
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

    # Check for multi-ICBM mode
    has_multi_icbm = result.samples and result.samples[0].icbm_positions

    # Build ICBM paths
    icbm_paths: Dict[str, Tuple[List[float], List[float]]] = {}
    if has_multi_icbm:
        icbm_names = sorted(result.samples[0].icbm_positions.keys())
        for name in icbm_names:
            xs: List[float] = []
            ys: List[float] = []
            for sample in result.samples:
                pos = sample.icbm_positions.get(name)
                if pos:
                    xs.append(pos[0])
                    ys.append(pos[1])
                else:
                    xs.append(math.nan)
                    ys.append(math.nan)
            icbm_paths[name] = (xs, ys)
    else:
        icbm_x = [sample.icbm_position[0] for sample in result.samples]
        icbm_y = [sample.icbm_position[1] for sample in result.samples]
        icbm_paths["ICBM"] = (icbm_x, icbm_y)

    interceptor_paths, _ = _interceptor_time_series(result)
    interceptor_names = list(interceptor_paths.keys())

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

    fig, ax = plt.subplots(figsize=(12, 6) if has_multi_icbm else (10, 5))
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Altitude (m)")
    if has_multi_icbm:
        ax.set_title(f"Multi-ICBM Intercept Simulation ({result.partial_success_count}/{result.total_icbm_count} destroyed)")
    else:
        ax.set_title("ICBM Intercept Simulation (Animation)")
    ax.grid(True, linewidth=0.2)

    # Create ICBM lines
    icbm_lines: Dict[str, any] = {}
    icbm_cmap = plt.get_cmap("Set1")
    for idx, (name, (xs, ys)) in enumerate(icbm_paths.items()):
        color = icbm_cmap(idx % icbm_cmap.N) if has_multi_icbm else "#4D4D4D"
        icbm_lines[name], = ax.plot([], [], color=color, linewidth=2.0, label=name)

    interceptor_lines: Dict[str, any] = {}
    for name in interceptor_names:
        report = result.interceptor_reports[name]
        style = _interceptor_style(report.config_name)
        style.setdefault("linewidth", 2.0)
        interceptor_lines[name], = ax.plot([], [], label=f"{name}", **style)
    # Create decoy lines with consolidated legend entry
    decoy_lines: Dict[int, any] = {}
    decoy_color = "#888888"  # Gray for all decoys
    for display_idx, decoy_id in enumerate(decoy_id_order, start=1):
        # Only first decoy gets a legend label
        label = f"Decoys ({len(decoy_id_order)})" if display_idx == 1 else None
        decoy_lines[decoy_id], = ax.plot(
            [], [], linestyle="--", linewidth=0.6, color=decoy_color, alpha=0.7, label=label
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

    # Add defense site markers
    site_markers = []
    if result.defense_sites:
        for site in result.defense_sites:
            marker = ax.scatter(
                [site.position[0]], [0.0],
                color="green", marker="^", s=150,
                label=f"{site.name}",
                zorder=5,
            )
            site_markers.append(marker)

    # Improved legend with multiple columns for many items
    num_legend_items = (
        len(icbm_paths) +
        len(interceptor_lines) +
        (1 if decoy_id_order else 0) +
        len(intercept_markers) +
        len(site_markers)
    )
    ncol = 2 if num_legend_items > 8 else 1
    ax.legend(loc="upper right", fontsize=7, ncol=ncol, framealpha=0.9)

    def init() -> List[any]:
        for line in icbm_lines.values():
            line.set_data([], [])
        for line in interceptor_lines.values():
            line.set_data([], [])
        for line in decoy_lines.values():
            line.set_data([], [])
        return [*icbm_lines.values(), *interceptor_lines.values(), *decoy_lines.values(), *intercept_markers, *site_markers]

    def update(frame: int) -> List[any]:
        upto = frame + 1
        for name, line in icbm_lines.items():
            xs, ys = icbm_paths[name]
            line.set_data(xs[:upto], ys[:upto])
        for name, line in interceptor_lines.items():
            xs, ys = interceptor_paths[name]
            line.set_data(xs[:upto], ys[:upto])
        for decoy_id, line in decoy_lines.items():
            xs, ys = decoy_paths[decoy_id]
            line.set_data(xs[:upto], ys[:upto])
        return [*icbm_lines.values(), *interceptor_lines.values(), *decoy_lines.values(), *intercept_markers, *site_markers]

    # Establish axis limits so trajectories are visible from the first frame.
    all_x: List[float] = []
    all_y: List[float] = []
    for xs, ys in icbm_paths.values():
        all_x.extend(x for x in xs if not math.isnan(x))
        all_y.extend(y for y in ys if not math.isnan(y))
    for name, (xs, ys) in interceptor_paths.items():
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

    def bounded_probability(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("probability must be a number") from exc
        if not 0.0 <= parsed <= 1.0:
            raise argparse.ArgumentTypeError("probability must be between 0 and 1")
        return parsed

    def non_negative_int(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("value must be an integer") from exc
        if parsed < 0:
            raise argparse.ArgumentTypeError("value must be >= 0")
        return parsed

    kwdefaults = simulate_icbm_intercept.__kwdefaults__ or {}
    default_confusion = kwdefaults.get("decoy_confusion_probability", 0.1)
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
        "--workers",
        "--threads",
        type=non_negative_int,
        default=1,
        help="number of worker processes for Monte Carlo runs (0 = all cores)",
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
    parser.add_argument(
        "--decoy-confusion-probability",
        type=bounded_probability,
        default=default_confusion,
        help=(
            "baseline probability (0-1) that seekers initially lock onto a decoy; "
            "the GBI layer halves this value while the THAAD layer adds 0.03 up to 0.12"
        ),
    )
    parser.add_argument(
        "--legacy-atmosphere",
        action="store_true",
        help="use legacy exponential atmosphere model instead of US Standard 1976",
    )
    parser.add_argument(
        "--earth-radius",
        type=float,
        default=6_371_000.0,
        help="earth radius in meters for gravity falloff (default 6371000)",
    )
    parser.add_argument(
        "--adaptive-dt",
        action="store_true",
        help="enable adaptive time stepping",
    )
    parser.add_argument(
        "--adaptive-dt-min",
        type=float,
        default=None,
        help="minimum adaptive time step in seconds (default dt * 0.2)",
    )
    parser.add_argument(
        "--adaptive-dt-max",
        type=float,
        default=None,
        help="maximum adaptive time step in seconds (default dt)",
    )
    parser.add_argument(
        "--mirv-count",
        type=int,
        default=1,
        help="number of MIRV warheads released by each ICBM (default 1)",
    )
    parser.add_argument(
        "--mirv-release-time",
        type=float,
        default=None,
        help="time after launch to release MIRVs (seconds)",
    )
    parser.add_argument(
        "--mirv-spread-velocity",
        type=float,
        default=120.0,
        help="relative spread speed for MIRV warheads (m/s)",
    )
    # ICBM/Decoy overrides
    parser.add_argument(
        "--decoy-count",
        type=int,
        default=None,
        help="override decoy count for all ICBMs",
    )
    parser.add_argument(
        "--decoy-release-time",
        type=float,
        default=None,
        help="override decoy release time in seconds after launch",
    )
    parser.add_argument(
        "--decoy-spread-velocity",
        type=float,
        default=None,
        help="override decoy spread velocity in m/s",
    )
    parser.add_argument(
        "--decoy-drag-multiplier",
        type=float,
        default=None,
        help="override decoy drag multiplier",
    )
    parser.add_argument(
        "--decoy-mass-fraction",
        type=float,
        default=None,
        help="override decoy mass fraction",
    )
    parser.add_argument(
        "--warhead-mass-fraction",
        type=float,
        default=None,
        help="override warhead mass fraction",
    )
    parser.add_argument(
        "--warhead-drag-multiplier",
        type=float,
        default=None,
        help="override warhead drag multiplier",
    )
    parser.add_argument(
        "--icbm-rcs",
        type=float,
        default=None,
        help="override ICBM radar cross section in m^2",
    )
    parser.add_argument(
        "--warhead-ballistic-coeff",
        type=float,
        default=None,
        help="override warhead ballistic coefficient in kg/m^2",
    )
    # Multi-ICBM arguments
    parser.add_argument(
        "--icbm-count",
        type=int,
        default=1,
        help="number of ICBMs to simulate (default 1)",
    )
    parser.add_argument(
        "--icbm-spacing",
        type=float,
        default=50000.0,
        help="horizontal spacing in meters between ICBM launch positions (default 50000)",
    )
    parser.add_argument(
        "--icbm-launch-interval",
        type=float,
        default=0.0,
        help="time interval in seconds between ICBM launches (default 0 for simultaneous)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["mixed", "standard", "fast", "heavy", "decoy-heavy"],
        default=None,
        help="ICBM scenario: 'mixed' uses one of each variant, or specify a single variant type",
    )
    parser.add_argument(
        "--icbm-variants",
        type=str,
        default=None,
        help="comma-separated list of ICBM variants (e.g., 'standard,fast,heavy,decoy-heavy')",
    )
    # Multi-site defense arguments
    parser.add_argument(
        "--defense-sites",
        type=int,
        default=1,
        help="number of defense sites (default 1)",
    )
    parser.add_argument(
        "--site-spacing",
        type=float,
        default=500000.0,
        help="horizontal spacing in meters between defense sites (default 500000)",
    )
    parser.add_argument(
        "--batteries-per-site",
        type=int,
        default=2,
        help="number of batteries per defense site (default 2: GBI + THAAD)",
    )
    parser.add_argument(
        "--launchers-per-battery",
        type=int,
        default=1,
        help="number of launchers per battery (default 1)",
    )
    parser.add_argument(
        "--interceptors-per-launcher",
        type=int,
        default=4,
        help="number of interceptors per launcher (default 4)",
    )
    # Interceptor layer arguments
    parser.add_argument(
        "--gbi-speed-cap",
        type=float,
        default=5400.0,
        help="GBI interceptor speed cap in m/s (default 5400.0)",
    )
    parser.add_argument(
        "--thaad-speed-cap",
        type=float,
        default=5000.0,
        help="THAAD interceptor speed cap in m/s (default 5000.0)",
    )
    parser.add_argument(
        "--gbi-launch-delay",
        type=float,
        default=120.0,
        help="GBI launch delay in seconds (default 120.0)",
    )
    parser.add_argument(
        "--thaad-launch-delay",
        type=float,
        default=340.0,
        help="THAAD launch delay in seconds (default 340.0)",
    )
    parser.add_argument(
        "--gbi-guidance-gain",
        type=float,
        default=0.88,
        help="GBI guidance gain (default 0.88)",
    )
    parser.add_argument(
        "--thaad-guidance-gain",
        type=float,
        default=0.68,
        help="THAAD guidance gain (default 0.68)",
    )
    parser.add_argument(
        "--gbi-damping-gain",
        type=float,
        default=0.055,
        help="GBI damping gain (default 0.055)",
    )
    parser.add_argument(
        "--thaad-damping-gain",
        type=float,
        default=0.11,
        help="THAAD damping gain (default 0.11)",
    )
    parser.add_argument(
        "--gbi-intercept-distance",
        type=float,
        default=96000.0,
        help="GBI intercept distance in meters (default 96000.0)",
    )
    parser.add_argument(
        "--thaad-intercept-distance",
        type=float,
        default=180000.0,
        help="THAAD intercept distance in meters (default 180000.0)",
    )
    parser.add_argument(
        "--gbi-max-accel",
        type=float,
        default=60.0,
        help="GBI maximum acceleration in m/s^2 (default 60.0)",
    )
    parser.add_argument(
        "--thaad-max-accel",
        type=float,
        default=155.0,
        help="THAAD maximum acceleration in m/s^2 (default 155.0)",
    )
    parser.add_argument(
        "--gbi-guidance-noise",
        type=float,
        default=0.03,
        help="GBI guidance noise standard deviation in degrees (default 0.03)",
    )
    parser.add_argument(
        "--thaad-guidance-noise",
        type=float,
        default=0.035,
        help="THAAD guidance noise standard deviation in degrees (default 0.035)",
    )
    parser.add_argument(
        "--gbi-reacquisition-rate",
        type=float,
        default=0.018,
        help="GBI decoy reacquisition rate (default 0.018)",
    )
    parser.add_argument(
        "--thaad-reacquisition-rate",
        type=float,
        default=0.06,
        help="THAAD decoy reacquisition rate (default 0.06)",
    )
    parser.add_argument(
        "--gbi-max-flight-time",
        type=float,
        default=1200.0,
        help="GBI maximum flight time in seconds (default 1200.0)",
    )
    parser.add_argument(
        "--thaad-max-flight-time",
        type=float,
        default=800.0,
        help="THAAD maximum flight time in seconds (default 800.0)",
    )

    # Radar and discrimination arguments
    parser.add_argument(
        "--no-discrimination",
        dest="use_discrimination",
        action="store_false",
        help="disable physics-based radar discrimination (falls back to probabilistic confusion)",
    )
    parser.set_defaults(use_discrimination=True)
    parser.add_argument(
        "--radar-range",
        type=float,
        default=4000_000.0,
        help="maximum radar detection range in meters (default 4000 km)",
    )
    parser.add_argument(
        "--radar-pos",
        type=float,
        nargs=2,
        default=None,
        help="radar (x, altitude) position; defaults to interceptor site",
    )
    parser.add_argument(
        "--radar-update-rate",
        type=float,
        default=10.0,
        help="radar tracking update rate in Hz (default 10.0)",
    )
    parser.add_argument(
        "--radar-min-rcs",
        type=float,
        default=0.001,
        help="minimum detectable RCS at max range in m^2 (default 0.001)",
    )
    parser.add_argument(
        "--radar-position-noise",
        type=float,
        default=50.0,
        help="radar position measurement noise standard deviation in meters (default 50.0)",
    )
    parser.add_argument(
        "--radar-velocity-noise",
        type=float,
        default=5.0,
        help="radar velocity measurement noise standard deviation in m/s (default 5.0)",
    )
    parser.add_argument(
        "--radar-track-threshold",
        type=int,
        default=3,
        help="number of consecutive detections needed for stable track (default 3)",
    )
    parser.add_argument(
        "--radar-antenna-height",
        type=float,
        default=30.0,
        help="radar antenna height above ground in meters (default 30.0)",
    )
    args = parser.parse_args()

    base_rng = random.Random(args.seed) if args.seed is not None else None

    # Build ICBM configs based on scenario, variants, or count
    icbm_configs: Optional[List[ICBMConfig]] = None
    
    if args.scenario is not None:
        # Use scenario-based configuration
        if args.scenario == "mixed":
            icbm_configs = create_mixed_salvo(
                spacing=args.icbm_spacing,
                launch_interval=args.icbm_launch_interval,
            )
        else:
            # Single variant type, use icbm_count to determine how many
            count = max(1, args.icbm_count)
            variants = [args.scenario] * count
            icbm_configs = create_mixed_salvo(
                spacing=args.icbm_spacing,
                launch_interval=args.icbm_launch_interval,
                variants=variants,
            )
    elif args.icbm_variants is not None:
        # Parse comma-separated variants
        variants = [v.strip() for v in args.icbm_variants.split(",")]
        icbm_configs = create_mixed_salvo(
            spacing=args.icbm_spacing,
            launch_interval=args.icbm_launch_interval,
            variants=variants,
        )
    elif args.icbm_count > 1:
        # Legacy mode: create generic ICBMs
        icbm_configs = []
        for i in range(args.icbm_count):
            icbm_configs.append(
                ICBMConfig(
                    name=f"ICBM-{i + 1}",
                    start_position=(i * args.icbm_spacing, 0.0),
                    launch_time=i * args.icbm_launch_interval,
                    mirv_count=max(1, args.mirv_count),
                    mirv_release_time=args.mirv_release_time,
                    mirv_spread_velocity=args.mirv_spread_velocity,
                )
            )

    # Apply any ICBM/MIRV overrides to all configurations
    if icbm_configs is not None:
        icbm_overrides = {}
        if args.mirv_count > 1 or args.mirv_release_time is not None:
            icbm_overrides.update({
                "mirv_count": max(1, args.mirv_count),
                "mirv_release_time": args.mirv_release_time,
                "mirv_spread_velocity": args.mirv_spread_velocity,
            })
        
        # Add the new ICBM/Decoy overrides
        if args.decoy_count is not None:
            icbm_overrides["decoy_count"] = args.decoy_count
        if args.decoy_release_time is not None:
            icbm_overrides["decoy_release_time"] = args.decoy_release_time
        if args.decoy_spread_velocity is not None:
            icbm_overrides["decoy_spread_velocity"] = args.decoy_spread_velocity
        if args.decoy_drag_multiplier is not None:
            icbm_overrides["decoy_drag_multiplier"] = args.decoy_drag_multiplier
        if args.decoy_mass_fraction is not None:
            icbm_overrides["decoy_mass_fraction"] = args.decoy_mass_fraction
        if args.warhead_mass_fraction is not None:
            icbm_overrides["warhead_mass_fraction"] = args.warhead_mass_fraction
        if args.warhead_drag_multiplier is not None:
            icbm_overrides["warhead_drag_multiplier"] = args.warhead_drag_multiplier
        if args.icbm_rcs is not None:
            icbm_overrides["rcs"] = args.icbm_rcs
        if args.warhead_ballistic_coeff is not None:
            icbm_overrides["warhead_ballistic_coeff"] = args.warhead_ballistic_coeff

        if icbm_overrides:
            adjusted_configs: List[ICBMConfig] = []
            for cfg in icbm_configs:
                cfg_data = dict(cfg.__dict__)
                cfg_data.update(icbm_overrides)
                adjusted_configs.append(ICBMConfig(**cfg_data))
            icbm_configs = adjusted_configs

    # Build defense sites if multi-site mode
    defense_sites: Optional[List[DefenseSiteConfig]] = None
    if args.defense_sites > 1 or args.launchers_per_battery > 1 or args.interceptors_per_launcher != 4:
        defense_sites = []
        base_site_x = 3_800_000.0

        for site_idx in range(args.defense_sites):
            site_x = base_site_x + site_idx * args.site_spacing
            site_position = (site_x, 0.0)

            batteries: List[BatteryConfig] = []

            # Create GBI battery
            gbi_launchers = tuple(
                LauncherConfig(interceptor_count=args.interceptors_per_launcher)
                for _ in range(args.launchers_per_battery)
            )
            gbi_template = InterceptorConfig(
                name="GBI",
                site=site_position,
                launch_delay=args.gbi_launch_delay,
                engage_altitude_min=120_000.0,
                engage_altitude_max=1_200_000.0,
                engage_range_min=0.0,  # Removed restrictive minimum - allow engagement at any range
                engage_range_max=6_000_000.0,
                speed_cap=args.gbi_speed_cap,
                guidance_gain=args.gbi_guidance_gain,
                damping_gain=args.gbi_damping_gain,
                intercept_distance=args.gbi_intercept_distance,
                max_accel=args.gbi_max_accel,
                guidance_noise_std_deg=args.gbi_guidance_noise,
                confusion_probability=min(0.15, args.decoy_confusion_probability * 0.5),
                reacquisition_rate=args.gbi_reacquisition_rate,
                max_flight_time=args.gbi_max_flight_time,
                salvo_count=1,
                salvo_interval=2.0,
            )
            batteries.append(BatteryConfig(
                name="GBI",
                interceptor_template=gbi_template,
                launchers=gbi_launchers,
            ))

            # Create THAAD battery if we have at least 2 batteries per site
            if args.batteries_per_site >= 2:
                thaad_launchers = tuple(
                    LauncherConfig(interceptor_count=args.interceptors_per_launcher)
                    for _ in range(args.launchers_per_battery)
                )
                thaad_template = InterceptorConfig(
                    name="THAAD",
                    site=site_position,
                    launch_delay=args.thaad_launch_delay,
                    engage_altitude_min=20_000.0,
                    engage_altitude_max=220_000.0,
                    engage_range_min=0.0,
                    engage_range_max=800_000.0,  # Increased from 260km to 800km to allow more engagement opportunities
                    speed_cap=args.thaad_speed_cap,
                    guidance_gain=args.thaad_guidance_gain,
                    damping_gain=args.thaad_damping_gain,
                    intercept_distance=args.thaad_intercept_distance,
                    max_accel=args.thaad_max_accel,
                    guidance_noise_std_deg=args.thaad_guidance_noise,
                    confusion_probability=min(0.12, args.decoy_confusion_probability + 0.03),
                    reacquisition_rate=args.thaad_reacquisition_rate,
                    max_flight_time=args.thaad_max_flight_time,
                    depends_on="GBI",
                    dependency_grace_period=45.0,
                    salvo_count=1,
                    salvo_interval=1.5,
                )
                batteries.append(BatteryConfig(
                    name="THAAD",
                    interceptor_template=thaad_template,
                    launchers=thaad_launchers,
                ))

            defense_sites.append(DefenseSiteConfig(
                name=f"Site-{site_idx + 1}",
                position=site_position,
                batteries=tuple(batteries),
            ))

    # Use provided interceptor site or default to 3.8 Mm
    interceptor_site = (3_800_000.0, 0.0)

    # Configure radar if requested
    radar_config = None
    if args.use_discrimination:
        radar_pos = tuple(args.radar_pos) if args.radar_pos else interceptor_site
        radar_config = RadarConfig(
            name="UEWR",
            position=radar_pos,
            max_range=args.radar_range,
            update_rate=args.radar_update_rate,
            min_rcs_at_max_range=args.radar_min_rcs,
            position_noise_std=args.radar_position_noise,
            velocity_noise_std=args.radar_velocity_noise,
            track_initiation_threshold=args.radar_track_threshold,
            antenna_height=args.radar_antenna_height,
        )

    result = simulate_icbm_intercept(
        rng=base_rng,
        gbi_salvo_count=args.gbi_salvo,
        gbi_salvo_interval=args.gbi_salvo_interval,
        thaad_salvo_count=args.thaad_salvo,
        thaad_salvo_interval=args.thaad_salvo_interval,
        decoy_confusion_probability=args.decoy_confusion_probability,
        use_standard_atmosphere=not args.legacy_atmosphere,
        earth_radius=args.earth_radius,
        adaptive_time_step=args.adaptive_dt,
        adaptive_dt_min=args.adaptive_dt_min,
        adaptive_dt_max=args.adaptive_dt_max,
        mirv_count=max(1, args.mirv_count),
        mirv_release_time=args.mirv_release_time,
        mirv_spread_velocity=args.mirv_spread_velocity,
        decoy_count=args.decoy_count if args.decoy_count is not None else 3,
        decoy_release_time=args.decoy_release_time,
        decoy_spread_velocity=args.decoy_spread_velocity if args.decoy_spread_velocity is not None else 280.0,
        decoy_drag_multiplier=args.decoy_drag_multiplier if args.decoy_drag_multiplier is not None else 4.0,
        decoy_mass_fraction=args.decoy_mass_fraction if args.decoy_mass_fraction is not None else 0.04,
        warhead_mass_fraction=args.warhead_mass_fraction if args.warhead_mass_fraction is not None else 0.35,
        warhead_drag_multiplier=args.warhead_drag_multiplier if args.warhead_drag_multiplier is not None else 0.6,
        icbm_rcs=args.icbm_rcs,
        warhead_ballistic_coeff=args.warhead_ballistic_coeff,
        icbm_configs=icbm_configs,
        defense_sites=defense_sites,
        use_discrimination=args.use_discrimination,
        radar_config=radar_config,
        gbi_speed_cap=args.gbi_speed_cap,
        thaad_speed_cap=args.thaad_speed_cap,
        gbi_launch_delay=args.gbi_launch_delay,
        thaad_launch_delay=args.thaad_launch_delay,
        gbi_guidance_gain=args.gbi_guidance_gain,
        thaad_guidance_gain=args.thaad_guidance_gain,
        gbi_damping_gain=args.gbi_damping_gain,
        thaad_damping_gain=args.thaad_damping_gain,
        gbi_intercept_distance=args.gbi_intercept_distance,
        thaad_intercept_distance=args.thaad_intercept_distance,
        gbi_max_accel=args.gbi_max_accel,
        thaad_max_accel=args.thaad_max_accel,
        gbi_guidance_noise=args.gbi_guidance_noise,
        thaad_guidance_noise=args.thaad_guidance_noise,
        gbi_reacquisition_rate=args.gbi_reacquisition_rate,
        thaad_reacquisition_rate=args.thaad_reacquisition_rate,
    )
    # Print summary
    if result.total_icbm_count > 1:
        print(f"Multi-ICBM Simulation: {result.total_icbm_count} ICBMs")
        print(f"  Overall success (all destroyed): {result.overall_success}")
        print(f"  ICBMs destroyed: {result.partial_success_count}/{result.total_icbm_count}")
        print()
        for name, outcome in sorted(result.icbm_outcomes.items()):
            status = "DESTROYED" if outcome.destroyed else ("IMPACTED" if outcome.impacted else "ESCAPED")
            details = []
            if outcome.destroyed and outcome.destroyed_by:
                details.append(f"by {outcome.destroyed_by} at t={outcome.intercept_time:.1f}s")
            if outcome.impacted and outcome.impact_time:
                details.append(f"at t={outcome.impact_time:.1f}s")
            if outcome.decoys_deployed > 0:
                details.append(f"{outcome.decoys_deployed} decoys deployed")
            detail_str = f" ({', '.join(details)})" if details else ""
            print(f"  {name}: {status}{detail_str}")
    else:
        print(_summarize(result))

    print(f"Sample count: {len(result.samples)} | Intercept success: {result.intercept_success}")
    for name in sorted(result.interceptor_reports.keys()):
        report = result.interceptor_reports[name]
        print("  " + _describe_interceptor(name, report, result.icbm_impact_time))
    
    # Print engagement statistics
    print()
    print(_engagement_statistics(result))

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
                "decoy_confusion_probability": args.decoy_confusion_probability,
                "use_standard_atmosphere": not args.legacy_atmosphere,
                "earth_radius": args.earth_radius,
                "adaptive_time_step": args.adaptive_dt,
                "adaptive_dt_min": args.adaptive_dt_min,
                "adaptive_dt_max": args.adaptive_dt_max,
                "mirv_count": max(1, args.mirv_count),
                "mirv_release_time": args.mirv_release_time,
                "mirv_spread_velocity": args.mirv_spread_velocity,
                "decoy_count": args.decoy_count if args.decoy_count is not None else 3,
                "decoy_release_time": args.decoy_release_time,
                "decoy_spread_velocity": args.decoy_spread_velocity if args.decoy_spread_velocity is not None else 280.0,
                "decoy_drag_multiplier": args.decoy_drag_multiplier if args.decoy_drag_multiplier is not None else 4.0,
                "decoy_mass_fraction": args.decoy_mass_fraction if args.decoy_mass_fraction is not None else 0.04,
                "warhead_mass_fraction": args.warhead_mass_fraction if args.warhead_mass_fraction is not None else 0.35,
                "warhead_drag_multiplier": args.warhead_drag_multiplier if args.warhead_drag_multiplier is not None else 0.6,
                "icbm_rcs": args.icbm_rcs,
                "warhead_ballistic_coeff": args.warhead_ballistic_coeff,
                "icbm_configs": icbm_configs,
                "defense_sites": defense_sites,
                "use_discrimination": args.use_discrimination,
                "radar_config": radar_config,
                "gbi_speed_cap": args.gbi_speed_cap,
                "thaad_speed_cap": args.thaad_speed_cap,
                "gbi_launch_delay": args.gbi_launch_delay,
                "thaad_launch_delay": args.thaad_launch_delay,
                "gbi_guidance_gain": args.gbi_guidance_gain,
                "thaad_guidance_gain": args.thaad_guidance_gain,
                "gbi_damping_gain": args.gbi_damping_gain,
                "thaad_damping_gain": args.thaad_damping_gain,
                "gbi_intercept_distance": args.gbi_intercept_distance,
                "thaad_intercept_distance": args.thaad_intercept_distance,
                "gbi_max_accel": args.gbi_max_accel,
                "thaad_max_accel": args.thaad_max_accel,
                "gbi_guidance_noise": args.gbi_guidance_noise,
                "thaad_guidance_noise": args.thaad_guidance_noise,
                "gbi_reacquisition_rate": args.gbi_reacquisition_rate,
                "thaad_reacquisition_rate": args.thaad_reacquisition_rate,
            },
            details=details_list,
            max_workers=args.workers,
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
