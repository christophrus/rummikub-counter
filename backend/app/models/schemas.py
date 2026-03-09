"""
Pydantic-Schemas für die Rummikub-Stein-Erkennung.
"""

from pydantic import BaseModel


class DetectedTile(BaseModel):
    """Ein einzelner erkannter Rummikub-Stein."""

    number: int | None = None
    confidence: float = 0.0
    is_joker: bool = False
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


class AnalysisResult(BaseModel):
    """Ergebnis der Bildanalyse."""

    tiles: list[DetectedTile] = []
    total_score: int = 0
    tile_count: int = 0
    processing_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
