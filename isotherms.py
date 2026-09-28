"""Isotherm depths: how far down the water is still warmer than a threshold.

**D26**, the depth of the 26 °C isotherm, is the companion to TCHP. TCHP says
how much heat a column holds above 26 °C; this says how far down that warm layer
reaches. The distinction matters to a cyclone: a storm mixes the upper ocean as
it passes, so a thin warm layer is churned away and the storm starves, while a
thick one keeps feeding it.

**D20** is the conventional marker for the thermocline -- the boundary between
the warm surface ocean and the cold deep ocean.

One rule matters more than the rest: where the isotherm does not exist, the
answer is **NaN, never 0**. Zero would say "the crossing is exactly at the sea
surface", which is a different and false claim. Three cases give no value:

  * land -- the input is NaN all the way down;
  * the surface is already colder than the target, so the isotherm lies above
    the water rather than in it;
  * the column stays warmer than the target down to its deepest wet level
    (shallow shelf water), so the crossing would be below the seabed.

This is the same distinction TCHP draws between land (NaN) and "no water above
26 °C" (a real zero) -- but here there is no legitimate zero at all.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

D26_REFERENCE_C = 26.0
D20_REFERENCE_C = 20.0

ISOTHERM_UNITS = "m"


def _vertical_dim(da: xr.DataArray) -> str:
    for name in ("depth", "elevation"):
        if name in da.dims:
            return name
    raise ValueError(f"temperature needs a depth dimension, got dims {da.dims}")


def isotherm_depth(temperature: xr.DataArray, target: float) -> xr.DataArray:
    """Depth in metres where each column first falls through `target` °C.

    `temperature` is potential temperature in °C with a depth dimension (metres,
    positive down, ascending); the result drops that dimension. Temperature is
    taken as linear between levels, so the crossing is interpolated rather than
    snapped to the nearest level -- the levels are up to ~100 m apart at depth,
    and snapping would quantise the answer into visible steps.

    Where a column crosses the target more than once (a temperature inversion),
    the shallowest crossing is returned, which is the standard definition.
    """
    depth_dim = _vertical_dim(temperature)
    z = np.asarray(temperature[depth_dim].values, dtype=np.float64)
    if z.size < 2 or np.any(np.diff(z) <= 0) or z[0] < 0:
        raise ValueError("depth must be at least two ascending, non-negative levels")

    da = temperature.transpose(depth_dim, ...)
    excess = np.asarray(da.values, dtype=np.float32) - np.float32(target)

    upper, lower = excess[:-1], excess[1:]
    shape = (-1,) + (1,) * (excess.ndim - 1)
    dz = np.diff(z).reshape(shape)
    top = z[:-1].reshape(shape)

    # A crossing is a segment that starts at or above the target and ends below
    # it. Requiring both ends finite keeps the seabed out: the segment below the
    # deepest wet level is NaN, so it can never be chosen.
    both = np.isfinite(upper) & np.isfinite(lower)
    crosses = both & (upper >= 0) & (lower < 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = upper / (upper - lower)

    # inf where there is no crossing, so the per-column minimum picks the
    # shallowest real one and leaves untouched columns as inf.
    candidates = np.where(crosses, top + fraction * dz, np.inf)
    shallowest = candidates.min(axis=0)

    depth = np.where(np.isfinite(shallowest), shallowest, np.nan)

    result = da.isel({depth_dim: 0}, drop=True).copy(data=depth.astype(np.float64))
    result.name = f"d{int(target)}"
    result.attrs = {
        "long_name": f"Depth of the {target:g} °C isotherm",
        "units": ISOTHERM_UNITS,
        "reference_temperature_C": target,
        "derived_from": "sea_water_potential_temperature",
    }
    return result


def d26_depth(temperature: xr.DataArray) -> xr.DataArray:
    """Depth of the 26 °C isotherm -- the base of the cyclone-fuel layer."""
    return isotherm_depth(temperature, D26_REFERENCE_C)


def d20_depth(temperature: xr.DataArray) -> xr.DataArray:
    """Depth of the 20 °C isotherm -- the conventional thermocline marker."""
    return isotherm_depth(temperature, D20_REFERENCE_C)
