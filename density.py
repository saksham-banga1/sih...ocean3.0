"""Seawater density, for the isopycnals on a T-S diagram.

A T-S diagram without density contours is a scatter plot; with them it is a
water-mass diagram, because a water mass is a body of water of roughly constant
density formed at the surface somewhere and spread at depth. The contours are
what let the Persian Gulf and Red Sea outflows be read as separate limbs rather
than as noise.

This implements the one-atmosphere International Equation of State of Seawater
(UNESCO 1983, "EOS-80"), which gives density from practical salinity and
temperature at the surface. Sigma-theta is that density computed from POTENTIAL
temperature, minus 1000 -- and the model's `thetao` is potential temperature
(`sea_water_potential_temperature`), so the pairing here is the correct one.

The polynomial's own published check values are asserted in
scripts/test_ts_diagram.py, so an error in transcription fails a test rather
than quietly bending every contour on screen.
"""

from __future__ import annotations

import numpy as np

# Density of pure water (SMOW), kg/m^3, as a polynomial in temperature.
_PURE_WATER = (
    999.842594,
    6.793952e-2,
    -9.095290e-3,
    1.001685e-4,
    -1.120083e-6,
    6.536332e-9,
)

# Salinity terms: linear in S, then S^1.5, then S^2.
_A = (0.824493, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
_B = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
_C = 4.8314e-4


def density_at_surface(salinity, temperature):
    """Seawater density at one atmosphere, kg/m^3 (EOS-80).

    salinity is practical salinity (the model's `so`, published as 1e-3);
    temperature is degrees Celsius. Both may be arrays.
    """
    s = np.asarray(salinity, dtype=float)
    t = np.asarray(temperature, dtype=float)

    rho_w = (
        _PURE_WATER[0]
        + _PURE_WATER[1] * t
        + _PURE_WATER[2] * t**2
        + _PURE_WATER[3] * t**3
        + _PURE_WATER[4] * t**4
        + _PURE_WATER[5] * t**5
    )
    a = _A[0] + _A[1] * t + _A[2] * t**2 + _A[3] * t**3 + _A[4] * t**4
    b = _B[0] + _B[1] * t + _B[2] * t**2

    # S^1.5 is undefined for negative salinity; the model never produces it, but
    # clamping keeps a stray fill value from becoming a NaN contour.
    s = np.maximum(s, 0.0)
    return rho_w + a * s + b * s**1.5 + _C * s**2


def sigma_theta(salinity, potential_temperature):
    """Potential density anomaly, kg/m^3: density at the surface minus 1000."""
    return density_at_surface(salinity, potential_temperature) - 1000.0


def sigma_theta_grid(
    salinity_range: tuple[float, float],
    temperature_range: tuple[float, float],
    steps: int = 60,
) -> dict:
    """Sigma-theta over a T-S rectangle, for drawing isopycnals.

    Computed here rather than in the browser so the equation of state has one
    implementation, not two that can drift apart.
    """
    salinities = np.linspace(salinity_range[0], salinity_range[1], steps)
    temperatures = np.linspace(temperature_range[0], temperature_range[1], steps)
    grid = sigma_theta(salinities[None, :], temperatures[:, None])
    return {
        "salinities": [round(float(v), 4) for v in salinities],
        "temperatures": [round(float(v), 4) for v in temperatures],
        # sigma[temperature index][salinity index]
        "sigma": [[round(float(v), 4) for v in row] for row in grid],
    }
