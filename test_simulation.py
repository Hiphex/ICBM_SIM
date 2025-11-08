import math

import pytest

from simulation import simulate_icbm_intercept


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


def test_simulate_icbm_intercept_rejects_nonpositive_dt():
    with pytest.raises(ValueError, match="time step dt must be positive"):
        simulate_icbm_intercept(dt=0.0)
