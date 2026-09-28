"""
Glider Explorer: what the model holds along a planned underwater path.

A glider flies a sawtooth -- surface, dive, surface -- across a route while
recording the water it passes through. There is no glider reporting anywhere
near this basin (checked against the 1,283 platforms in the Copernicus global
in-situ feed: none inside 62-96E, 4-24N, and none in the wider northern Indian
Ocean either), so this module does not pretend one exists.

What it does instead: you give it a route and a dive profile, and it reports
what the ocean model already says is at each point along that route. The path
is a **plan**. Every value is the **model**. Neither is dressed as telemetry
from a vehicle, and nothing here is generated -- no synthetic temperatures, no
battery, no speed, no "live" anything.

Three rules, inherited from app/services/sections.py, whose great-circle path
and bilinear sampler this reuses rather than reimplements:

  * The route is a real great circle and the distances are real haversine
    kilometres, so a leg's length is its length.
  * Values are bilinearly interpolated from the model's own cells, and stay
    null where any corner of that interpolation is land. The coastline and the
    seafloor are the model's, not a shape invented here.
  * Depth is **snapped to a real model level**, never interpolated between two.
    A glider asked to fly at 137 m is reported against the model level that
    actually holds a value, and both numbers are returned -- the depth asked
    for and the level answered from. Section data follows the same rule, and
    the animal context already reports its `model_depth_m` this way.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from app.services.ocean_model import (
    _coord_name,
    _depth_dim,
    load_dataset,
    resolve_nc_variable,
)
from app.services.sections import great_circle_path, initial_bearing

# A mission is one interpolation per variable per sample. These caps keep a
# pathological request from pinning the process, and match sections.py.
MIN_SAMPLES = 2
MAX_SAMPLES = 400
MAX_WAYPOINTS = 12
MAX_CYCLES = 20


class MissionError(ValueError):
    """The request cannot produce a mission track."""


def dive_profile(fractions: np.ndarray, max_depth: float, cycles: int) -> np.ndarray:
    """A sawtooth: surface, down to `max_depth`, back to the surface, `cycles` times.

    This is the shape of a glider's flight, not a simulation of one. Nothing
    here models buoyancy, pitch or speed -- it is the geometry the operator
    asked for, and it is described that way wherever it is served.
    """
    if cycles < 1:
        raise MissionError("a mission needs at least one dive cycle")
    phase = np.mod(fractions * cycles, 1.0)
    # 0 -> 1 -> 0 across each cycle.
    triangle = np.where(phase < 0.5, phase * 2.0, (1.0 - phase) * 2.0)
    return triangle * max_depth


def route(waypoints: list[tuple[float, float]], samples: int
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Chain great-circle legs through every waypoint.

    Samples are shared out in proportion to each leg's length, so a long leg
    is not sampled as coarsely as a short one. Returns
    (lats, lons, cumulative_km, total_km).
    """
    if len(waypoints) < 2:
        raise MissionError("a route needs at least two waypoints")
    if len(waypoints) > MAX_WAYPOINTS:
        raise MissionError(f"a route takes at most {MAX_WAYPOINTS} waypoints")
    if not MIN_SAMPLES <= samples <= MAX_SAMPLES:
        raise MissionError(f"samples must be between {MIN_SAMPLES} and {MAX_SAMPLES}")

    # Measure every leg first, so samples can be shared out by length.
    legs = []
    for start, end in zip(waypoints, waypoints[1:]):
        if start == end:
            raise MissionError(f"waypoints {start} and {end} are the same point")
        _, _, _, leg_km = great_circle_path(start, end, 2)
        legs.append(leg_km)
    total_km = float(sum(legs))
    if total_km <= 0:
        raise MissionError("the route has no length")

    lats: list[float] = []
    lons: list[float] = []
    dists: list[float] = []
    travelled = 0.0
    for index, ((start, end), leg_km) in enumerate(zip(zip(waypoints, waypoints[1:]), legs)):
        share = max(2, int(round(samples * leg_km / total_km)))
        leg_lats, leg_lons, leg_dists, _ = great_circle_path(start, end, share)
        # Drop the repeated waypoint where one leg meets the next.
        first = 0 if index == 0 else 1
        lats.extend(leg_lats[first:].tolist())
        lons.extend(leg_lons[first:].tolist())
        dists.extend((leg_dists[first:] + travelled).tolist())
        travelled += leg_km

    return np.asarray(lats), np.asarray(lons), np.asarray(dists), total_km


def sample_mission(
    variables: list[str],
    date_str: str,
    waypoints: list[tuple[float, float]],
    *,
    samples: int = 160,
    max_depth: float = 500.0,
    cycles: int = 4,
    round_to: int = 4,
) -> dict:
    """The model along a planned glider route, one point per sample."""
    if not variables:
        raise MissionError("name at least one variable to sample")
    if max_depth <= 0:
        raise MissionError("max_depth must be greater than zero")
    if cycles > MAX_CYCLES:
        raise MissionError(f"at most {MAX_CYCLES} dive cycles")

    lats, lons, distances, total_km = route(waypoints, samples)
    fractions = distances / total_km if total_km else np.zeros_like(distances)
    wanted_depths = dive_profile(fractions, max_depth, cycles)

    lat_da = xr.DataArray(lats, dims="sample")
    lon_da = xr.DataArray(lons, dims="sample")

    per_variable: dict[str, list] = {}
    model_depths: np.ndarray | None = None
    level_values: np.ndarray | None = None

    for variable in variables:
        dataset = load_dataset(variable, date_str)
        array = dataset[resolve_nc_variable(variable)]
        if "time" in array.dims:
            array = array.isel(time=0)

        depth_name = _depth_dim(array)
        if depth_name is None:
            raise MissionError(
                f"{variable!r} is a 2D field with no depth axis, so a glider cannot fly through it")

        levels = np.asarray(array[depth_name].values, dtype=float)
        usable = levels[levels <= max_depth]
        if usable.size == 0:
            raise MissionError(
                f"no model level within {max_depth:g} m -- the shallowest is {levels.min():.3f} m")

        # Snap each requested depth to the nearest real level, and remember
        # which level that was. Interpolating between levels would report a
        # resolution the model does not have.
        nearest = np.abs(usable[None, :] - wanted_depths[:, None]).argmin(axis=1)
        snapped = usable[nearest]
        if model_depths is None:
            model_depths, level_values = snapped, usable
        elif not np.array_equal(snapped, model_depths):
            # Different products sit on different vertical grids; say so
            # rather than quietly reporting one variable's levels for another.
            raise MissionError(
                f"{variable!r} is on a different set of model levels from the others; "
                "request it on its own")

        lat_name = _coord_name(array, "latitude", "lat")
        lon_name = _coord_name(array, "longitude", "lon")
        # Depth is selected, not interpolated -- `snapped` already holds real
        # levels, so this is a bilinear sample in lat/lon at a chosen level,
        # exactly as sections.py does it. Out-of-grid comes back NaN.
        sampled = array.interp(
            {lat_name: lat_da, lon_name: lon_da,
             depth_name: xr.DataArray(snapped, dims="sample")},
            method="linear",
        )
        values = np.asarray(sampled.values, dtype=float).ravel()
        if values.size != lats.size:
            raise MissionError(f"{variable!r} sampled {values.size} points for {lats.size} positions")
        per_variable[variable] = [
            None if not np.isfinite(v) else round(float(v), round_to) for v in values
        ]

    first = per_variable[variables[0]]
    in_water = sum(1 for v in first if v is not None)
    if in_water == 0:
        raise MissionError(
            "the whole route is land, below the seafloor, or outside the downloaded grid")

    legs = [
        {"from": {"lat": round(a[0], 4), "lon": round(a[1], 4)},
         "to": {"lat": round(b[0], 4), "lon": round(b[1], 4)},
         "bearing_deg": round(initial_bearing(a, b), 1)}
        for a, b in zip(waypoints, waypoints[1:])
    ]

    return {
        "waypoints": [{"lat": round(w[0], 4), "lon": round(w[1], 4)} for w in waypoints],
        "legs": legs,
        "latitudes": [round(float(v), 4) for v in lats],
        "longitudes": [round(float(v), 4) for v in lons],
        "distances_km": [round(float(v), 3) for v in distances],
        "requested_depths_m": [round(float(v), 2) for v in wanted_depths],
        "model_depths_m": [round(float(v), 3) for v in model_depths],
        "values": per_variable,
        "model_levels_m": [round(float(v), 3) for v in level_values],
        "total_distance_km": round(float(total_km), 3),
        "sample_count": int(lats.size),
        "samples_in_water": in_water,
        "max_depth_m": float(max_depth),
        "dive_cycles": int(cycles),
    }
