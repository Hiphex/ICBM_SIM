import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation import InterceptorConfig, run_monte_carlo, simulate_icbm_intercept


@pytest.mark.parametrize("invalid_dt", [0.0, -0.5])
def test_simulate_icbm_intercept_rejects_non_positive_dt(invalid_dt):
    with pytest.raises(ValueError) as exc_info:
        simulate_icbm_intercept(dt=invalid_dt)
    message = str(exc_info.value)
    assert "time step" in message.lower()
    assert "positive" in message.lower()


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


def _run_until_guard(dt: float) -> tuple[float, int]:
    with pytest.raises(RuntimeError) as exc_info:
        simulate_icbm_intercept(
            dt=dt,
            gravity=0.0,
            interceptors=[],
            decoy_count=0,
            decoy_release_time=None,
        )

    assert "safety iteration limit" in str(exc_info.value)

    tb = exc_info.value.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next

    frame_locals = tb.tb_frame.f_locals
    return frame_locals["time"], frame_locals["step_count"]


def test_guard_time_horizon_independent_of_dt():
    target_seconds = 50_000.0
    for dt in (0.25, 0.1, 5.0):
        time, step_count = _run_until_guard(dt)
        assert time == pytest.approx(target_seconds, abs=dt)
        expected_steps = math.ceil(target_seconds / dt)
        assert step_count == expected_steps
