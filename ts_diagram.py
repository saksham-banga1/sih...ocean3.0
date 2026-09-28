"""T-S diagrams: temperature against salinity down a model water column.

This is the plot that separates water masses. A map shows where water is; a
section shows how deep it goes; a T-S diagram shows *what it is*. Water formed
at the surface in one place keeps its temperature-salinity signature as it
spreads at depth, so in the Arabian Sea the Persian Gulf and Red Sea outflows
appear as distinct limbs rather than as a smooth curve.

Two correctness rules matter more than anything else here:

  * A T-S point is only meaningful when the temperature and the salinity come
    from the SAME level of the SAME cell. The two variables live in separate
    files, and ocean_model.sample_profile_at_point() drops land levels per
    variable -- so zipping its two outputs would pair a temperature at one depth
    with a salinity from another the moment either had a gap. Everything below
    pairs on the depth coordinate itself.
  * Sigma-theta is potential density, so it must be computed from POTENTIAL
    temperature. The model's `thetao` is exactly that
    (`sea_water_potential_temperature`), which is why no conversion appears.
"""

from __future__ import annotations

import numpy as np

from app.services import ocean_model
from app.services.density import sigma_theta

# A region scatter is one point per level per column; this caps the payload.
MAX_COLUMNS = 120
MAX_POINTS = 6000


class TSError(ValueError):
    """The request cannot produce a T-S diagram (off the grid, all land)."""


def _paired_column(temp_ds, salt_ds, lat: float, lon: float, max_depth: float):
    """(depths, temperatures, salinities, grid_lat, grid_lon) at one cell."""
    temperature = ocean_model.sample_profile_at_point(temp_ds, lat, lon, "sst")
    salinity = ocean_model.sample_profile_at_point(salt_ds, lat, lon, "salinity")
    if temperature is None or salinity is None:
        return None

    t_depths, t_values, t_lat, t_lon = temperature
    s_depths, s_values, s_lat, s_lon = salinity

    # The two products are published on the same grid, but pairing is done on
    # the depth values themselves rather than on position in the array, so a
    # level missing from one and present in the other cannot shift the pairing.
    common, t_index, s_index = np.intersect1d(t_depths, s_depths, return_indices=True)
    if common.size == 0:
        return None

    keep = common <= max_depth
    if not keep.any():
        return None

    return (
        common[keep],
        t_values[t_index][keep],
        s_values[s_index][keep],
        (t_lat, t_lon),
        (s_lat, s_lon),
    )


def _levels(depths, temperatures, salinities) -> list[dict]:
    densities = sigma_theta(salinities, temperatures)
    return [
        {
            "depth": round(float(d), 3),
            "temperature": round(float(t), 4),
            "salinity": round(float(s), 4),
            "sigma_theta": round(float(sig), 4),
        }
        for d, t, s, sig in zip(depths, temperatures, salinities, densities)
    ]


def column_diagram(date_str: str, lat: float, lon: float, *, max_depth: float = 2000.0) -> dict:
    """Every level of one model column as a (T, S, sigma-theta) point."""
    temp_ds = ocean_model.load_dataset("sst", date_str)
    salt_ds = ocean_model.load_dataset("salinity", date_str)

    paired = _paired_column(temp_ds, salt_ds, lat, lon, max_depth)
    if paired is None:
        raise TSError(
            "no water column there: the point is land, or outside the downloaded grid"
        )
    depths, temperatures, salinities, t_cell, s_cell = paired

    if t_cell != s_cell:
        # Never observed with these products, but pairing values from two
        # different cells would be a quiet fabrication, so it fails loudly.
        raise TSError(
            f"temperature and salinity resolved to different grid cells "
            f"({t_cell} vs {s_cell}); refusing to pair them"
        )

    return {
        "columns": [
            {
                "lat": round(t_cell[0], 4),
                "lon": round(t_cell[1], 4),
                "levels": _levels(depths, temperatures, salinities),
            }
        ],
        "point_count": int(depths.size),
    }


def region_diagram(
    date_str: str,
    bbox: tuple[float, float, float, float],
    *,
    columns: int = 36,
    max_depth: float = 2000.0,
) -> dict:
    """A scatter over a box: several columns, so water masses show as clusters."""
    if not 1 <= columns <= MAX_COLUMNS:
        raise TSError(f"columns must be between 1 and {MAX_COLUMNS}")

    temp_ds = ocean_model.load_dataset("sst", date_str)
    salt_ds = ocean_model.load_dataset("salinity", date_str)

    min_lon, min_lat, max_lon, max_lat = bbox
    side = max(1, int(round(columns**0.5)))
    lats = np.linspace(min_lat, max_lat, side)
    lons = np.linspace(min_lon, max_lon, side)

    out: list[dict] = []
    total = 0
    for lat in lats:
        for lon in lons:
            paired = _paired_column(temp_ds, salt_ds, float(lat), float(lon), max_depth)
            if paired is None:
                continue                      # land, or off the grid: skipped, not filled
            depths, temperatures, salinities, t_cell, s_cell = paired
            if t_cell != s_cell:
                continue
            levels = _levels(depths, temperatures, salinities)
            if total + len(levels) > MAX_POINTS:
                break
            total += len(levels)
            out.append({"lat": round(t_cell[0], 4), "lon": round(t_cell[1], 4), "levels": levels})

    if not out:
        raise TSError("no water columns in that box: all land, or outside the grid")

    return {"columns": out, "point_count": total}


# Below this many points every one is shown. A single column is 40 levels and
# each is a real measurement of the water: trimming its ends to tidy the axes
# would hide the surface or the seabed, which is not a trade worth making.
CLIP_MIN_POINTS = 100
CLIP_PERCENTILE = 1.0


def bounds(columns: list[dict]) -> dict:
    """Padded axis ranges for the plot, from the data actually returned.

    Across a box, a handful of extreme points -- a river plume at 13 PSU beside
    open ocean at 36 -- stretch the axes so far that everything else is crushed
    into a corner. Once there are enough points the axes are set from the 1st
    and 99th percentiles instead, and the count left outside is returned so the
    panel can say so on screen. Nothing is dropped from `columns`: the data is
    all still there, and only the view is narrowed.
    """
    temperatures = np.array(
        [level["temperature"] for column in columns for level in column["levels"]]
    )
    salinities = np.array(
        [level["salinity"] for column in columns for level in column["levels"]]
    )
    sigmas = [level["sigma_theta"] for column in columns for level in column["levels"]]

    def span(values, fraction=0.04):
        if values.size >= CLIP_MIN_POINTS:
            low = float(np.percentile(values, CLIP_PERCENTILE))
            high = float(np.percentile(values, 100.0 - CLIP_PERCENTILE))
        else:
            low, high = float(values.min()), float(values.max())
        width = (high - low) or 1.0
        return round(low - width * fraction, 4), round(high + width * fraction, 4)

    temperature_range = span(temperatures)
    salinity_range = span(salinities)
    outside = int(
        np.count_nonzero(
            (temperatures < temperature_range[0])
            | (temperatures > temperature_range[1])
            | (salinities < salinity_range[0])
            | (salinities > salinity_range[1])
        )
    )

    return {
        "temperature_range": list(temperature_range),
        "salinity_range": list(salinity_range),
        # The density range describes the water itself, so it stays the full
        # range even when the view is narrowed.
        "sigma_range": [round(min(sigmas), 4), round(max(sigmas), 4)],
        "points_outside": outside,
    }
