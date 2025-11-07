import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation import simulate_icbm_intercept


@pytest.mark.parametrize("invalid_dt", [0.0, -0.5])
def test_simulate_icbm_intercept_rejects_non_positive_dt(invalid_dt):
    with pytest.raises(ValueError) as exc_info:
        simulate_icbm_intercept(dt=invalid_dt)
    message = str(exc_info.value)
    assert "time step" in message.lower()
    assert "positive" in message.lower()
