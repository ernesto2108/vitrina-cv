"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest  # noqa: TC002


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that are slow (integration / real-fixture tests).",
    )
