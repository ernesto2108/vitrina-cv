"""Semantic engines package — Strategy pattern for furniture/element detection.

Analogous to vitrina_cv.engines (GeometryEngine, ADR-008) but for the semantic
classification track (ADR-002, ADR-004, run 11).
"""

from vitrina_cv.engines.semantic.base import SemanticEngine, get_semantic_engine

__all__ = ["SemanticEngine", "get_semantic_engine"]
