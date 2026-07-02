"""CV engines package — Strategy pattern for geometry extraction (ADR-008)."""

from vitrina_cv.engines.base import GeometryEngine, get_engine

__all__ = ["GeometryEngine", "get_engine"]
