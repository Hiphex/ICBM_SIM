import json
import math
import os
import random

import pytest

from simulation import (
    BatteryConfig,
    DefenseSiteConfig,
    ICBMConfig,
    InterceptorConfig,
    LauncherConfig,
    _interceptor_time_series,
    run_monte_carlo,
    simulate_icbm_intercept,
)


def test_run_monte_carlo_details_drawn_parameters_are_json_safe():
    custom_interceptor = InterceptorConfig(
        name="Custom",
        site=(1.0, 2.0),
        launch_delay=130.0,
        engage_altitude_min=10_000.0,
        engage_altitude_max=600_000.0,
        engage_range_min=0.0,
        engage_range_max=500_000.0,
        speed_cap=4_600.0,
        guidance_gain=0.72,
        damping_gain=0.12,
        intercept_distance=1_500.0,
        max_accel=55.0,
        guidance_noise_std_deg=0.04,
        confusion_probability=0.08,
        reacquisition_rate=0.02,
        max_flight_time=500.0,
        depends_on=None,
        dependency_grace_period=0.0,
        salvo_count=1,
        salvo_interval=0.0,
    )

    details = []
    run_monte_carlo(
        1,
        base_kwargs={"interceptors": [custom_interceptor]},
        details=details,
    )

    assert details, "Expected details to contain at least one entry"
    drawn_parameters = details[0]["drawn_parameters"]

    json.dumps(drawn_parameters)

    interceptors = drawn_parameters.get("interceptors")
    assert isinstance(interceptors, list)
    assert interceptors and isinstance(interceptors[0], dict)


@pytest.mark.skipif(
    os.cpu_count() is None or os.cpu_count() < 2,
    reason="requires multiple CPU cores",
)
def test_run_monte_carlo_parallel_matches_serial():
    base_kwargs = {
        "decoy_count": 0,
        "decoy_release_time": None,
        "use_standard_atmosphere": False,
        "earth_radius": 1.0e9,
    }

    serial_details: list[dict[str, object]] = []
    serial = run_monte_carlo(
        4,
        seed=12345,
        base_kwargs=base_kwargs,
        details=serial_details,
        max_workers=1,
    )

    parallel_details: list[dict[str, object]] = []
    parallel = run_monte_carlo(
        4,
        seed=12345,
        base_kwargs=base_kwargs,
        details=parallel_details,
        max_workers=2,
    )

    assert serial.runs == parallel.runs
    assert serial.successes == parallel.successes
    assert serial.impacts == parallel.impacts
    assert serial.timeouts == parallel.timeouts
    assert serial.intercept_times == parallel.intercept_times
    assert serial.miss_distances == parallel.miss_distances
    assert serial.decoy_intercepts == parallel.decoy_intercepts
    assert serial.layer_primary_kills == parallel.layer_primary_kills
    assert serial.layer_decoy_hits == parallel.layer_decoy_hits
    assert serial_details == parallel_details


def _layered_stack_for_thaad_intercept() -> list[InterceptorConfig]:
    gbi = InterceptorConfig(
        name="GBI",
        site=(3_800_000.0, 0.0),
        launch_delay=120.0,
        engage_altitude_min=120_000.0,
        engage_altitude_max=1_200_000.0,
        engage_range_min=400_000.0,
        engage_range_max=6_000_000.0,
        speed_cap=1_500.0,
        guidance_gain=0.05,
        damping_gain=0.02,
        intercept_distance=1.0,
        max_accel=10.0,
        guidance_noise_std_deg=0.0,
        confusion_probability=0.0,
        reacquisition_rate=0.0,
        max_flight_time=400.0,
    )
    thaad = InterceptorConfig(
        name="THAAD",
        site=(3_800_000.0, 0.0),
        launch_delay=340.0,
        engage_altitude_min=20_000.0,
        engage_altitude_max=220_000.0,
        engage_range_min=0.0,
        engage_range_max=800_000.0,
        speed_cap=5_800.0,
        guidance_gain=0.85,
        damping_gain=0.11,
        intercept_distance=500_000.0,
        max_accel=180.0,
        guidance_noise_std_deg=0.0,
        confusion_probability=0.0,
        reacquisition_rate=0.08,
        max_flight_time=1_200.0,
        depends_on="GBI",
        dependency_grace_period=45.0,
    )
    return [gbi, thaad]


def test_thaad_layer_catches_primary_when_gbi_fails():
    result = simulate_icbm_intercept(
        interceptors=_layered_stack_for_thaad_intercept(),
        decoy_count=0,
        decoy_release_time=None,
        use_standard_atmosphere=False,
        earth_radius=1.0e9,
    )

    assert result.intercept_success is True
    assert result.intercept_target_label == "primary"

    gbi_report = result.interceptor_reports["GBI"]
    thaad_report = result.interceptor_reports["THAAD"]

    assert gbi_report.success is False
    assert thaad_report.success is True
    assert thaad_report.target_label == "primary"
    assert thaad_report.intercept_time == result.intercept_time
    assert thaad_report.intercept_position is not None


def test_interceptor_time_series_includes_thaad_path():
    result = simulate_icbm_intercept(
        interceptors=_layered_stack_for_thaad_intercept(),
        decoy_count=0,
        decoy_release_time=None,
        use_standard_atmosphere=False,
        earth_radius=1.0e9,
    )

    thaad_report = result.interceptor_reports["THAAD"]

    series, intercept_indices = _interceptor_time_series(result)
    thaad_series = series["THAAD"]
    intercept_idx = intercept_indices["THAAD"]

    assert intercept_idx is not None

    xs, ys = thaad_series
    assert intercept_idx < len(xs)
    assert intercept_idx < len(ys)

    x_at_intercept = xs[intercept_idx]
    y_at_intercept = ys[intercept_idx]

    assert math.isfinite(x_at_intercept)
    assert math.isfinite(y_at_intercept)

    assert math.isclose(
        x_at_intercept,
        thaad_report.intercept_position[0],
        rel_tol=1e-6,
        abs_tol=1e-3,
    )
    assert math.isclose(
        y_at_intercept,
        thaad_report.intercept_position[1],
        rel_tol=1e-6,
        abs_tol=1e-3,
    )


# ============================================================================
# Multi-ICBM Tests
# ============================================================================


def test_multi_icbm_basic_simulation():
    """Test that multiple ICBMs can be simulated."""
    icbm_configs = [
        ICBMConfig(name="ICBM-1", start_position=(0.0, 0.0)),
        ICBMConfig(name="ICBM-2", start_position=(50000.0, 0.0)),
    ]

    result = simulate_icbm_intercept(
        icbm_configs=icbm_configs,
        decoy_count=0,
        decoy_release_time=None,
        rng=random.Random(42),
    )

    assert result.total_icbm_count == 2
    assert len(result.icbm_outcomes) == 2
    assert "ICBM-1" in result.icbm_outcomes
    assert "ICBM-2" in result.icbm_outcomes

    # Check that trajectory samples contain multi-ICBM data
    assert result.samples
    first_sample = result.samples[0]
    assert "ICBM-1" in first_sample.icbm_positions
    assert "ICBM-2" in first_sample.icbm_positions


def test_multi_icbm_partial_success():
    """Test that partial success is tracked correctly."""
    # Create 3 ICBMs with staggered launches
    icbm_configs = [
        ICBMConfig(name="ICBM-1", start_position=(0.0, 0.0), launch_time=0.0),
        ICBMConfig(name="ICBM-2", start_position=(100000.0, 0.0), launch_time=10.0),
        ICBMConfig(name="ICBM-3", start_position=(200000.0, 0.0), launch_time=20.0),
    ]

    result = simulate_icbm_intercept(
        icbm_configs=icbm_configs,
        decoy_count=0,
        decoy_release_time=None,
        rng=random.Random(123),
    )

    assert result.total_icbm_count == 3

    # Count outcomes
    destroyed = sum(1 for o in result.icbm_outcomes.values() if o.destroyed)
    impacted = sum(1 for o in result.icbm_outcomes.values() if o.impacted)
    escaped = sum(1 for o in result.icbm_outcomes.values() if o.escaped)

    assert result.partial_success_count == destroyed
    assert destroyed + impacted + escaped == 3

    # Overall success is True only if ALL destroyed
    assert result.overall_success == (destroyed == 3)


def test_multi_icbm_outcomes_have_correct_fields():
    """Test that ICBMOutcome has all expected fields."""
    icbm_configs = [
        ICBMConfig(name="TestICBM", start_position=(0.0, 0.0), decoy_count=2),
    ]

    result = simulate_icbm_intercept(
        icbm_configs=icbm_configs,
        rng=random.Random(456),
    )

    outcome = result.icbm_outcomes["TestICBM"]

    assert outcome.name == "TestICBM"
    assert isinstance(outcome.destroyed, bool)
    assert isinstance(outcome.impacted, bool)
    assert isinstance(outcome.escaped, bool)
    # Exactly one of these should be True
    assert sum([outcome.destroyed, outcome.impacted, outcome.escaped]) == 1


# ============================================================================
# Multi-Site Defense Tests
# ============================================================================


def test_defense_site_configuration():
    """Test that defense sites can be configured with batteries and launchers."""
    gbi_template = InterceptorConfig(
        name="GBI",
        site=(3_800_000.0, 0.0),
        launch_delay=120.0,
        engage_altitude_min=120_000.0,
        engage_altitude_max=1_200_000.0,
        engage_range_min=400_000.0,
        engage_range_max=6_000_000.0,
        speed_cap=5400.0,
        guidance_gain=0.88,
        damping_gain=0.055,
        intercept_distance=96_000.0,
        max_accel=60.0,
        guidance_noise_std_deg=0.0,
        confusion_probability=0.0,
        reacquisition_rate=0.02,
        max_flight_time=900.0,
    )

    site = DefenseSiteConfig(
        name="Site-Alpha",
        position=(3_800_000.0, 0.0),
        batteries=(
            BatteryConfig(
                name="GBI-Battery",
                interceptor_template=gbi_template,
                launchers=(
                    LauncherConfig(interceptor_count=4),
                    LauncherConfig(interceptor_count=4),
                ),
            ),
        ),
    )

    result = simulate_icbm_intercept(
        defense_sites=[site],
        decoy_count=0,
        decoy_release_time=None,
        rng=random.Random(789),
    )

    # Should have created interceptors from the site
    assert len(result.interceptor_reports) > 0

    # Check that interceptor reports include site info
    for report in result.interceptor_reports.values():
        assert report.site_name == "Site-Alpha"
        assert report.battery_name == "GBI-Battery"


def test_multi_site_defense():
    """Test multiple defense sites working together."""
    gbi_template = InterceptorConfig(
        name="GBI",
        site=(0.0, 0.0),  # Will be overridden by site position
        launch_delay=120.0,
        engage_altitude_min=120_000.0,
        engage_altitude_max=1_200_000.0,
        engage_range_min=0.0,
        engage_range_max=6_000_000.0,
        speed_cap=5400.0,
        guidance_gain=0.88,
        damping_gain=0.055,
        intercept_distance=96_000.0,
        max_accel=60.0,
        guidance_noise_std_deg=0.0,
        confusion_probability=0.0,
        reacquisition_rate=0.02,
        max_flight_time=900.0,
    )

    sites = [
        DefenseSiteConfig(
            name="Site-1",
            position=(3_500_000.0, 0.0),
            batteries=(
                BatteryConfig(
                    name="GBI",
                    interceptor_template=gbi_template,
                    launchers=(LauncherConfig(interceptor_count=2),),
                ),
            ),
        ),
        DefenseSiteConfig(
            name="Site-2",
            position=(4_000_000.0, 0.0),
            batteries=(
                BatteryConfig(
                    name="GBI",
                    interceptor_template=gbi_template,
                    launchers=(LauncherConfig(interceptor_count=2),),
                ),
            ),
        ),
    ]

    result = simulate_icbm_intercept(
        defense_sites=sites,
        decoy_count=0,
        decoy_release_time=None,
        rng=random.Random(101112),
    )

    # Should have interceptors from both sites
    site_names = {r.site_name for r in result.interceptor_reports.values()}
    assert "Site-1" in site_names
    assert "Site-2" in site_names

    # Defense sites should be recorded in the result
    assert len(result.defense_sites) == 2


def test_interceptor_target_icbm_tracking():
    """Test that interceptors track which ICBM they target."""
    icbm_configs = [
        ICBMConfig(name="ICBM-Alpha", start_position=(0.0, 0.0)),
        ICBMConfig(name="ICBM-Beta", start_position=(100000.0, 0.0)),
    ]

    result = simulate_icbm_intercept(
        icbm_configs=icbm_configs,
        decoy_count=0,
        decoy_release_time=None,
        rng=random.Random(131415),
    )

    # Check that launched interceptors have target assignments
    for report in result.interceptor_reports.values():
        if report.launch_time is not None:
            # Launched interceptors should have a target ICBM
            assert report.target_icbm_name is not None
            assert report.target_icbm_name in ["ICBM-Alpha", "ICBM-Beta"]


def test_single_icbm_backward_compatibility():
    """Test that single ICBM mode still works with legacy parameters."""
    result = simulate_icbm_intercept(
        decoy_count=0,
        decoy_release_time=None,
        rng=random.Random(161718),
    )

    # Should work as before
    assert result.total_icbm_count == 1
    assert "ICBM-1" in result.icbm_outcomes
    assert result.overall_success == result.intercept_success
