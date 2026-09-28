"""Vertical sections: the model along a line, from the surface downwards.

A map answers "where"; a section answers "how deep". Sampling one variable down
a transect is what makes the thermocline, the Arabian Sea oxygen minimum zone
and the freshwater lid of the Bay of Bengal visible as *shapes* rather than as a
colour that changes when you drag the depth slider.

Three rules keep this honest:

  * The path is a real great circle, and the distances reported are real
    haversine kilometres -- not a straight line in lat/lon, which would
    misstate both the route and its length.
  * Values are bilinearly interpolated from the model's own cells. Where any
    corner of the interpolation is land, the result stays NaN and is served as
    null. The seafloor and the coastline are therefore the model's, not a shape
    this module invented.
  * Every level returned is a real model level. The vertical axis is uneven
    (0.494 m ... 1062 m ...) and is sent as-is rather than resampled onto
    round numbers that would imply resolution the model does not have.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from app.services.ocean_model import (
    _coord_name,
    _depth_dim,
    load_dataset,
    resolve_nc_variable,
)

EARTH_RADIUS_KM = 6371.0088

# A section is cheap per sample but not free: each one is an interpolation over
# every level. This cap keeps a pathological request from pinning the process.
MAX_SAMPLES = 400
MIN_SAMPLES = 2


class SectionError(ValueError):
    """The request cannot produce a section (2D field, degenerate path, no water)."""


def great_circle_path(
    start: tuple[float, float], end: tuple[float, float], samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """`samples` points along the great circle from start to end, inclusive.

    Returns (lats, lons, distances_km, total_km). Interpolating in lat/lon
    instead would drift off the true path and, at this basin's width, report a
    length several kilometres wrong.
    """
    lat1, lon1 = math.radians(start[0]), math.radians(start[1])
    lat2, lon2 = math.radians(end[0]), math.radians(end[1])

    # Haversine central angle between the endpoints.
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    angle = 2 * math.asin(min(1.0, math.sqrt(h)))
    if angle == 0:
        raise SectionError("the two ends of the section are the same point")

    fractions = np.linspace(0.0, 1.0, samples)
    # Spherical linear interpolation along the arc.
    a = np.sin((1 - fractions) * angle) / np.sin(angle)
    b = np.sin(fractions * angle) / np.sin(angle)
    x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
    y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
    z = a * math.sin(lat1) + b * math.sin(lat2)

    lats = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    lons = np.degrees(np.arctan2(y, x))
    total_km = angle * EARTH_RADIUS_KM
    return lats, lons, fractions * total_km, total_km


def initial_bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Compass bearing at the start of the transect, in degrees from north."""
    lat1, lat2 = math.radians(start[0]), math.radians(end[0])
    dlon = math.radians(end[1] - start[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def sample_section(
    variable: str,
    date_str: str,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    samples: int = 120,
    max_depth: float = 2000.0,
    round_to: int = 4,
) -> dict:
    """The model down a transect: depths x samples, land and seafloor left null."""
    if not MIN_SAMPLES <= samples <= MAX_SAMPLES:
        raise SectionError(f"samples must be between {MIN_SAMPLES} and {MAX_SAMPLES}")

    dataset = load_dataset(variable, date_str)
    array = dataset[resolve_nc_variable(variable)]
    if "time" in array.dims:
        array = array.isel(time=0)

    depth_name = _depth_dim(array)
    if depth_name is None:
        raise SectionError(f"{variable!r} has no vertical axis, so it has no section")

    lat_name = _coord_name(array, "latitude", "lat")
    lon_name = _coord_name(array, "longitude", "lon")

    array = array.sel({depth_name: slice(0.0, max_depth)})
    if array[depth_name].size == 0:
        raise SectionError(f"no model levels within {max_depth:g} m")

    lats, lons, distances, total_km = great_circle_path(start, end, samples)

    # One vectorised bilinear interpolation for the whole curtain: xarray pairs
    # the lat/lon arrays because they share a dimension name, so this samples
    # along the path rather than over a lat x lon rectangle.
    sampled = array.interp(
        {
            lat_name: xr.DataArray(lats, dims="sample"),
            lon_name: xr.DataArray(lons, dims="sample"),
        },
        method="linear",
    )
    values = np.asarray(sampled.transpose(depth_name, "sample").values, dtype=float)

    finite = np.isfinite(values)
    if not finite.any():
        raise SectionError(
            "the whole transect is land or lies outside the downloaded grid"
        )

    depths = [round(float(d), 3) for d in np.asarray(array[depth_name].values)]
    rows = [
        [None if not np.isfinite(v) else round(float(v), round_to) for v in row]
        for row in values
    ]

    return {
        "depths": depths,
        "distances_km": [round(float(d), 3) for d in distances],
        "latitudes": [round(float(v), 4) for v in lats],
        "longitudes": [round(float(v), 4) for v in lons],
        "values": rows,
        "total_distance_km": round(float(total_km), 3),
        "bearing_deg": round(initial_bearing(start, end), 1),
        "water_fraction": round(float(finite.sum() / finite.size), 4),
        "min_value": round(float(np.nanmin(values)), round_to),
        "max_value": round(float(np.nanmax(values)), round_to),
    }
