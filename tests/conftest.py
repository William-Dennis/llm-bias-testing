"""Shared fixtures for slm-bias-testing tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def sample_df():
    """A small DataFrame mimicking CV screening results."""
    import pandas as pd

    return pd.DataFrame(
        {
            "key": ["a", "a", "a", "b", "b", "b"],
            "run": [0, 1, 2, 0, 1, 2],
            "score": [80, 85, 78, 60, 65, 55],
            "name": ["Alice", "Alice", "Alice", "Bob", "Bob", "Bob"],
            "university": ["Oxbridge", "Oxbridge", "Oxbridge", "Redbrick", "Redbrick", "Redbrick"],
            "a_levels": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        }
    )
