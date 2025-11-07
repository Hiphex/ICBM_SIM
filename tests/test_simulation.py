import json

from simulation import InterceptorConfig, run_monte_carlo


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
