import json
import math

from simulation import (
    InterceptorConfig,
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
