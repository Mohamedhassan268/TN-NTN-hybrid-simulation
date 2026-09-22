from __future__ import annotations

import pytest

from drl.device import meets_speedup_threshold


def test_speedup_rejects_invalid_wall_times():
    with pytest.raises(ValueError):
        meets_speedup_threshold(0.0, 1.0)
    with pytest.raises(ValueError):
        meets_speedup_threshold(1.0, 0.0)
