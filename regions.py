"""
Shared basin bounding boxes.

Single source of truth for both scripts/download_ocean_data.py (Copernicus model
downloads) and app/services/argo.py (Argo float observations), so model data and
observations always describe the same regions.

The boxes split the model rectangle the globe renders --
Cesium.Rectangle.fromDegrees(62, 4, 96, 24) in main.js -- in half at 78E, the
midpoint of the two basin bookmark camera targets (btnArabianSea at 68E,
btnBayOfBengal at 88E). Together they tile that rectangle with no gap or overlap.
"""

from __future__ import annotations

# (min_lon, min_lat, max_lon, max_lat)
# Cesium's Rectangle.fromDegrees(west, south, east, north) order, as used in main.js.
PRESET_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "arabian_sea": (62.0, 4.0, 78.0, 24.0),
    "bay_of_bengal": (78.0, 4.0, 96.0, 24.0),
    "full_domain": (62.0, 4.0, 96.0, 24.0),
}


def get_bbox(preset: str) -> tuple[float, float, float, float]:
    """Look up a preset box by name."""
    try:
        return PRESET_BBOXES[preset]
    except KeyError:
        raise ValueError(
            f"unknown region {preset!r} -- expected one of {sorted(PRESET_BBOXES)}"
        )
