import math
import random

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


def test_ballistic_impact_without_interceptors():
    rng = random.Random(12345)

    result = simulate_icbm_intercept(
        interceptors=[],
        decoy_count=0,
        decoy_release_time=None,
        rng=rng,
    )

    assert result.intercept_success is False
    assert result.icbm_impact_time is not None
    assert math.isfinite(result.icbm_impact_time)
