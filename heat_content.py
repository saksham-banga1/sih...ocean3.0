"""
Tropical cyclone heat potential (TCHP), calculated from model temperature.

TCHP is the heat stored in water warmer than 26 °C -- roughly the sea
temperature tropical cyclones need beneath them to form and keep going --
summed from the surface down through the water column:

    TCHP = rho * cp * integral( max(T(z) - 26, 0) dz )        [kJ/cm^2]

It is calculated here from Copernicus model potential temperature (thetao), so
it is a derived model quantity: not an observation, and not a forecast of any
cyclone.

Numerics. The model has 40 uneven levels (0.494 m, 1.541 m, ...). Temperature
is treated as linear between levels and each segment's excess above 26 °C is
integrated exactly -- including the part of a segment where it crosses 26 °C --
so the answer does not hinge on where the levels happen to fall. Above the
first level the water is taken as uniform, as a mixed surface layer is. A column
that stays above 26 °C down to its deepest wet level (shallow shelf water) is
integrated to that level: that is all the warm water there is.

Constants are fixed: rho = 1025 kg/m^3 and cp = 3991.868 J/(kg K), the TEOS-10
cp0. Density varies by well under 1% across the warm layer, and potential and
in-situ temperature differ by hundredths of a degree in the top 150 m -- both
far inside the model's own uncertainty.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

TCHP_REFERENCE_C = 26.0

SEAWATER_DENSITY = 1025.0            # kg m^-3
SEAWATER_CP = 3991.86795711963       # J kg^-1 K^-1, the TEOS-10 constant cp0

# 1 kJ/cm^2 = 1e3 J / 1e-4 m^2 = 1e7 J/m^2
_J_PER_M2_PER_KJ_PER_CM2 = 1e7

TCHP_UNITS = "kJ/cm2"


def _vertical_dim(da: xr.DataArray) -> str:
    for name in ("depth", "elevation"):
        if name in da.dims:
            return name
    raise ValueError(f"temperature needs a depth dimension, got dims {da.dims}")


def tropical_cyclone_heat_potential(temperature: xr.DataArray) -> xr.DataArray:
    """Heat above 26 °C in each water column, in kJ/cm^2.

    `temperature` is potential temperature in °C with a depth dimension (metres,
    positive down, ascending) and any horizontal dimensions; the result drops the
    depth dimension. A column with no water at all (land) is NaN, and one with no
    water above 26 °C is 0 -- the two must not be confused on a map.
    """
    depth_dim = _vertical_dim(temperature)
    z = np.asarray(temperature[depth_dim].values, dtype=np.float32)
    if z.size < 2 or np.any(np.diff(z) <= 0) or z[0] < 0:
        raise ValueError("depth must be at least two ascending, non-negative levels")

    da = temperature.transpose(depth_dim, ...)
    # float32 matches the source files and halves the transient memory of a
    # full-domain column integration.
    excess = np.asarray(da.values, dtype=np.float32) - np.float32(TCHP_REFERENCE_C)

    upper, lower = excess[:-1], excess[1:]
    dz = np.diff(z).reshape((-1,) + (1,) * (excess.ndim - 1))
    both = np.isfinite(upper) & np.isfinite(lower)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Where a segment crosses 26 °C: the share of it above the crossing.
        crossing = upper / (upper - lower)

    # The excess is linear within a segment, so its positive part integrates
    # exactly -- a trapezoid when both ends are warm, a triangle when it crosses.
    area = np.where(both & (upper >= 0) & (lower >= 0), 0.5 * (upper + lower) * dz, 0)
    area = np.where(both & (upper > 0) & (lower < 0), 0.5 * upper * crossing * dz, area)
    area = np.where(both & (upper < 0) & (lower > 0), 0.5 * lower * (1 - crossing) * dz, area)

    # Above the first level the water is taken as uniform, like a mixed layer.
    top = excess[0]
    surface = np.where(np.isfinite(top) & (top > 0), top * z[0], 0)

    wet = np.isfinite(excess).any(axis=0)
    joules_per_m2 = (surface + area.sum(axis=0)).astype(np.float64) * SEAWATER_DENSITY * SEAWATER_CP
    tchp = np.where(wet, joules_per_m2 / _J_PER_M2_PER_KJ_PER_CM2, np.nan)

    result = da.isel({depth_dim: 0}, drop=True).copy(data=tchp)
    result.name = "tchp"
    result.attrs = {
        "long_name": "Tropical cyclone heat potential",
        "units": TCHP_UNITS,
        "reference_temperature_C": TCHP_REFERENCE_C,
        "derived_from": "sea_water_potential_temperature",
    }
    return result
