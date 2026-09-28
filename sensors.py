"""
Unified in-situ sensor view: moored buoys + Argo floats in one array.

Field names deliberately mirror the hardcoded `inSituSensors` array in main.js
(id, name, place, lat, lon, obsTemp, modelTemp, obsSalinity, modelSalinity,
obsWave, modelWave) so the frontend can swap the fetch in without reworking its
render code.

Two provenance levels, tagged with `source` so the UI can distinguish them:

* ``placeholder`` -- the moored buoy records copied verbatim from main.js.
  INCOIS does not expose a public live feed we can call, so these stay static.
* ``argo_live``   -- real floats from the Argo GDAC, with obsTemp/obsSalinity
  read off the shallowest level of an actual profile.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from app.services import argo as argo_service
from app.services import ocean_model
from app.services.ocean_model import DatasetNotFoundError
from app.services.regions import PRESET_BBOXES

SOURCE_PLACEHOLDER = "placeholder"
SOURCE_ARGO_LIVE = "argo_live"

# Copied verbatim from the inSituSensors array in main.js (lines 330-333).
# INCOIS has no public live feed we can call, so these remain static and are
# tagged `placeholder` rather than presented as live observations.
MOORED_BUOYS: list[dict] = [
    {
        "id": "BD08-INCOIS", "name": "Moored Buoy 8",
        "place": "North Bay of Bengal (Off Odisha)",
        "lat": 18.2, "lon": 89.6,
        "obsTemp": 29.4, "modelTemp": 29.1,
        "obsSalinity": 32.8, "modelSalinity": 33.1,
        "obsWave": 2.1, "modelWave": 2.0,
    },
    {
        "id": "BD11-INCOIS", "name": "Moored Buoy 11",
        "place": "Central Bay of Bengal (Off Chennai)",
        "lat": 14.2, "lon": 83.5,
        "obsTemp": 28.6, "modelTemp": 28.5,
        "obsSalinity": 33.5, "modelSalinity": 33.4,
        "obsWave": 1.6, "modelWave": 1.5,
    },
    {
        "id": "AD02-INCOIS", "name": "Moored Buoy 2",
        "place": "Deep Arabian Sea (Off Goa/Mumbai)",
        "lat": 15.1, "lon": 69.1,
        "obsTemp": 27.9, "modelTemp": 26.8,
        "obsSalinity": 36.2, "modelSalinity": 35.8,
        "obsWave": 3.4, "modelWave": 2.7,
    },
    {
        # main.js has four buoys, not three -- keeping this one so swapping the
        # endpoint in does not silently drop a marker the globe already draws.
        "id": "CB01-INCOIS", "name": "Coastal Station",
        "place": "Kochi Coastal Waters (Kerala)",
        "lat": 10.0, "lon": 75.8,
        "obsTemp": 29.1, "modelTemp": 29.0,
        "obsSalinity": 34.7, "modelSalinity": 34.6,
        "obsWave": 1.1, "modelWave": 1.1,
    },
]

# Depth levels for synthetic buoy profiles: 0-2000 m in 50 m steps, matching the
# range and step of the depth slider in structure.html.
SYNTHETIC_DEPTHS = list(range(0, 2001, 50))

# Constants from calculateSubsurfaceValue() in main.js, kept identical so the
# backend and the frontend's own fallback agree.
DEEP_TEMP = 4.0
TEMP_SCALE_HEIGHT = 350.0
DEEP_SALINITY = 34.8
SALINITY_SCALE_HEIGHT = 400.0
WAVE_SCALE_HEIGHT = 15.0
WAVE_CUTOFF_DEPTH = 40.0


class SensorNotFoundError(LookupError):
    """No sensor matched the requested id."""


def subsurface_value(surface_value: float, depth: float, variable: str) -> float:
    """Port of calculateSubsurfaceValue() in main.js.

    Exponential decay from a surface observation towards a deep-water constant.
    This is a placeholder model, not physics -- it exists so buoy records, which
    only report at the surface, can still populate a depth profile.
    """
    if variable == "sst":
        return round(
            DEEP_TEMP + (surface_value - DEEP_TEMP) * math.exp(-depth / TEMP_SCALE_HEIGHT), 2
        )
    if variable == "salinity":
        return round(
            DEEP_SALINITY
            + (surface_value - DEEP_SALINITY) * math.exp(-depth / SALINITY_SCALE_HEIGHT),
            2,
        )
    if variable == "wave":
        if depth > WAVE_CUTOFF_DEPTH:
            return 0.0
        return round(surface_value * math.exp(-depth / WAVE_SCALE_HEIGHT), 2)
    raise ValueError(f"unknown variable {variable!r} -- expected sst, salinity or wave")


def _basin_for(lat: float, lon: float) -> str:
    """Human-readable basin name for an Argo float's position."""
    for name in ("bay_of_bengal", "arabian_sea"):
        min_lon, min_lat, max_lon, max_lat = PRESET_BBOXES[name]
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return name.replace("_", " ").title()
    return "Indian Ocean"


def _model_value_at(lat: float, lon: float, date: str, variable: str) -> float | None:
    """Surface model value at a position, or None if no file covers that day."""
    try:
        dataset = ocean_model.load_dataset(variable, date)
    except (DatasetNotFoundError, ValueError):
        return None
    return ocean_model.sample_at_point(dataset, lat, lon, 0.0, variable)


def _argo_to_sensor(observation: dict) -> dict:
    """Turn one bulk surface observation into a main.js-shaped sensor record."""
    lat, lon = observation["lat"], observation["lon"]
    day = observation["last_seen"][:10]

    return {
        "id": observation["id"],
        "name": f"Argo Float {observation['wmo']}",
        "place": f"{_basin_for(lat, lon)} (cycle {observation['cycle']})",
        "lat": lat,
        "lon": lon,
        "obsTemp": observation["temp"],
        # Real model value where a downloaded file covers this float's day and
        # position; None otherwise -- never faked to match the observation.
        "modelTemp": _model_value_at(lat, lon, day, "sst"),
        "obsSalinity": observation["salinity"],
        "modelSalinity": _model_value_at(lat, lon, day, "salinity"),
        # Argo floats carry no wave sensor, and no wave model is downloaded yet.
        "obsWave": None,
        "modelWave": None,
        "source": SOURCE_ARGO_LIVE,
        "cycle": observation["cycle"],
        # Real surfacing positions for the drift polyline, oldest first.
        "track": observation.get("track") or [],
    }


def list_sensors(
    region: str = "full_domain",
    *,
    include_argo: bool = True,
    include_buoys: bool = True,
    days: int = argo_service.DEFAULT_ACTIVE_DAYS,
    limit: int | None = None,
) -> list[dict]:
    """All active sensors in one array, buoys first then Argo floats."""
    if region not in PRESET_BBOXES:
        raise ValueError(
            f"unknown region {region!r} -- expected one of {sorted(PRESET_BBOXES)}"
        )

    min_lon, min_lat, max_lon, max_lat = PRESET_BBOXES[region]
    sensors: list[dict] = []

    if include_buoys:
        sensors.extend(
            {**buoy, "source": SOURCE_PLACEHOLDER}
            for buoy in MOORED_BUOYS
            if min_lon <= buoy["lon"] <= max_lon and min_lat <= buoy["lat"] <= max_lat
        )

    if include_argo:
        # One bulk request for the whole box, rather than a profile fetch per float.
        observations, provenance = argo_service.fetch_surface_observations(
            min_lon, min_lat, max_lon, max_lat, days=days, with_provenance=True
        )
        last_provenance.update(provenance)
        if limit is not None:
            observations = observations[:limit]
        sensors.extend(_argo_to_sensor(obs) for obs in observations)

    return sensors


class ModelDataUnavailableError(LookupError):
    """No model file covers this date, or the sensor sits outside the grid."""


def compare_at_depth(
    sensor_id: str, depth: float, date: str, variable: str = "sst"
) -> dict:
    """Model vs observation for one sensor at one depth -- the Model Variance badge.

    Model side: nearest grid cell in the downloaded Copernicus file.
    Observed side: the float's real measured profile, or for buoys the decay
    formula from main.js applied to their surface value.
    """
    if variable not in ("sst", "salinity"):
        raise ValueError(f"unknown variable {variable!r} -- expected 'sst' or 'salinity'")

    sensor = get_sensor_profile(sensor_id)  # raises SensorNotFoundError
    lat, lon = sensor["lat"], sensor["lon"]

    # --- model side -------------------------------------------------------
    try:
        dataset = ocean_model.load_dataset(variable, date)
    except DatasetNotFoundError as exc:
        raise ModelDataUnavailableError(
            f"no data for this date/region: no {variable} model file covers {date}"
        ) from exc

    model_value = ocean_model.sample_at_point(dataset, lat, lon, depth, variable)
    if model_value is None:
        raise ModelDataUnavailableError(
            f"no data for this date/region: {sensor['id']} at {lat}N {lon}E is outside "
            f"the downloaded {variable} grid for {date}, or that cell is land"
        )
    model_depth = ocean_model.get_selected_depth(dataset, depth)

    # --- observed side ----------------------------------------------------
    buoy = get_buoy(sensor_id)
    if buoy is not None:
        # Buoys report only at the surface, so derive with the main.js formula.
        surface = buoy["obsTemp"] if variable == "sst" else buoy["obsSalinity"]
        observed_value = subsurface_value(surface, depth, variable)
        observed_depth = float(depth)
    else:
        key = "temp" if variable == "sst" else "salinity"
        levels = [lv for lv in sensor["profile"] if lv.get(key) is not None]
        if not levels:
            raise ModelDataUnavailableError(
                f"{sensor['id']} has no {variable} measurements to compare"
            )
        nearest = min(levels, key=lambda lv: abs(lv["depth"] - depth))
        observed_value = nearest[key]
        observed_depth = nearest["depth"]

    return {
        "sensor_id": sensor["id"],
        "variable": variable,
        # Sign convention matches main.js renderTelemetryCard:
        #   const diff = (obsDepthTemp - modelDepthTemp)
        "model_value": round(float(model_value), 3),
        "observed_value": round(float(observed_value), 3),
        "difference": round(float(observed_value) - float(model_value), 3),
        "depth": float(depth),
        "date": date,
        # When the observation was actually taken. An Argo float's latest cycle
        # can be days older than the model date, so the difference is not purely
        # model error -- the caller needs to see both to judge it.
        "observed_time": sensor.get("time"),
        # The two sides rarely sit on exactly the requested depth; report where
        # each value actually came from so the comparison can be judged.
        "model_depth": round(model_depth, 3),
        "observed_depth": round(observed_depth, 3),
        "source": sensor["source"],
    }


# Collocation thresholds. Beyond these the comparison is still returned, just
# marked low-confidence -- hiding it would be worse than labelling it.
#
# Time: the model fields are DAILY MEANS, so +/-12 h is inherent to any match.
# 24 h is one full averaging window: past that, the timestep being compared is a
# different day's average than the one the float sampled.
MAX_TIME_SEPARATION_HOURS = 24.0

# Space: the grid is 1/12 degree, about 9.3 km per cell, so an open-ocean match
# is normally under ~7 km. 25 km is roughly two and a half cells -- enough slack
# for a coastal float whose nearest wet cell is a little offshore, but tight
# enough to catch a float being compared against genuinely different water.
MAX_DISTANCE_KM = 25.0


# Provenance of the most recent Argo fetch, so routers can report whether the
# data came straight from the GDAC or from the persistent real-data cache.
last_provenance: dict = {"source": "live", "last_synced_at": None}


class SyntheticProfileError(ValueError):
    """Profile validation was asked for on a sensor whose profile is generated."""


def _skill_metrics(observed: np.ndarray, modelled: np.ndarray) -> dict:
    """MAE, RMSE and bias over matched levels. Bias is observed - model.

    Same sign convention as compare_at_depth and the Model Variance badge:
    positive means the observation is warmer/saltier than the model.
    """
    residual = observed - modelled
    return {
        "mae": round(float(np.mean(np.abs(residual))), 4),
        "rmse": round(float(np.sqrt(np.mean(residual**2))), 4),
        "bias": round(float(np.mean(residual)), 4),
        "n": int(residual.size),
        "observed_mean": round(float(np.mean(observed)), 4),
        "model_mean": round(float(np.mean(modelled)), 4),
    }


def _parse_observed_time(value: str | None) -> datetime | None:
    """Argo timestamps arrive as 'YYYY-MM-DDTHH:MM:SS'; fall back to the date."""
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def compare_profile(
    sensor_id: str,
    date: str | None = None,
    *,
    max_levels: int | None = 200,
    max_time_separation_hours: float = MAX_TIME_SEPARATION_HOURS,
    max_distance_km: float = MAX_DISTANCE_KM,
) -> dict:
    """Validate a whole Argo cast against the model water column.

    The model is sampled at its nearest grid cell and linearly interpolated onto
    the float's own measurement depths. Only depths inside the model's vertical
    range are kept -- extrapolating past the deepest model level would invent
    values, so those Argo levels are dropped instead.

    `date` defaults to the day the float actually surfaced, which is the honest
    comparison; if that day was never downloaded, the nearest available is used
    and the offset is reported.
    """
    sensor = get_sensor_profile(sensor_id)  # raises SensorNotFoundError

    if sensor["source"] != SOURCE_ARGO_LIVE:
        raise SyntheticProfileError(
            f"{sensor['id']} has no measured profile -- its levels are generated by "
            f"the decay formula, so validating the model against them would be "
            f"circular. Profile validation is available for Argo floats only."
        )

    observed_levels = sensor["profile"]
    if not observed_levels:
        raise SyntheticProfileError(f"{sensor['id']} returned no measured levels")

    # --- temporal collocation ----------------------------------------------
    # The float's own timestamp picks the model timestep. A UI date is honoured
    # only when explicitly passed, and even then the separation is measured
    # against the observation, so a poor match cannot be hidden.
    observed_at = _parse_observed_time(sensor["time"])
    if observed_at is None and date is None:
        raise ModelDataUnavailableError(
            f"no data for this date/region: {sensor['id']} has no observation time "
            f"and no date was supplied"
        )

    if date is not None:
        # Explicit override: use that day, but still measure the real gap.
        model_day = datetime.strptime(date, "%Y-%m-%d").date()
        if model_day not in ocean_model.available_days("sst"):
            nearest_day = ocean_model.find_nearest_available_date("sst", date)
            if nearest_day is None:
                raise ModelDataUnavailableError(
                    "no data for this date/region: no model files are downloaded at all"
                )
            model_day = nearest_day
        model_instant = datetime(
            model_day.year, model_day.month, model_day.day,
            ocean_model.DAILY_MEAN_CENTRE_HOUR,
        )
    else:
        collocated = ocean_model.find_nearest_timestep("sst", observed_at)
        if collocated is None:
            raise ModelDataUnavailableError(
                "no data for this date/region: no model files are downloaded at all"
            )
        model_day, model_instant, _ = collocated

    model_date = str(model_day)
    requested = date or (sensor["time"] or "")[:10]

    time_separation_hours = (
        round(abs((model_instant - observed_at).total_seconds()) / 3600.0, 2)
        if observed_at
        else None
    )
    date_offset_days = (
        abs((model_day - observed_at.date()).days) if observed_at else 0
    )

    lat, lon = sensor["lat"], sensor["lon"]
    argo_depth = np.array([lv["depth"] for lv in observed_levels], dtype=float)

    matched: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    grid_lat = grid_lon = None
    model_depth_range: list[float] | None = None

    distance_km: float | None = None

    for variable, key in (("sst", "temp"), ("salinity", "salinity")):
        try:
            dataset = ocean_model.load_dataset(variable, model_date)
        except DatasetNotFoundError:
            continue

        # Spatial collocation: nearest cell that actually holds water. Snapping
        # blindly can land on land and drop coastal floats entirely.
        cell = ocean_model.find_nearest_valid_cell(dataset, lat, lon, variable)
        if cell is None:
            continue
        cell_lat, cell_lon, cell_km = cell

        column = ocean_model.sample_profile_at_point(dataset, cell_lat, cell_lon, variable)
        if column is None:
            continue

        model_depth, model_value, grid_lat, grid_lon = column
        distance_km = cell_km

        observed = np.array(
            [lv[key] if lv[key] is not None else np.nan for lv in observed_levels],
            dtype=float,
        )

        # Overlapping, valid levels only: inside the model's vertical range and
        # finite on the observed side. No extrapolation.
        overlap = (
            np.isfinite(observed)
            & (argo_depth >= model_depth.min())
            & (argo_depth <= model_depth.max())
        )
        if not overlap.any():
            continue

        interpolated = np.interp(argo_depth[overlap], model_depth, model_value)

        matched[f"depth_{variable}"] = argo_depth[overlap]
        matched[f"observed_{variable}"] = observed[overlap]
        matched[f"model_{variable}"] = interpolated
        metrics[variable] = _skill_metrics(observed[overlap], interpolated)
        model_depth_range = [round(float(model_depth.min()), 3),
                             round(float(model_depth.max()), 3)]

    if not metrics:
        raise ModelDataUnavailableError(
            f"no data for this date/region: no model column overlaps {sensor['id']} "
            f"at {lat}N {lon}E on {model_date} (outside the downloaded grid, on land, "
            f"or no depths in common)"
        )

    # --- build the paired level list ---------------------------------------
    # Temperature drives the shared depth axis; salinity is matched onto it when
    # present, so the frontend gets one array to plot from.
    primary = "sst" if "sst" in metrics else "salinity"
    depths = matched[f"depth_{primary}"]

    step = 1
    if max_levels and depths.size > max_levels:
        step = int(np.ceil(depths.size / max_levels))

    def at(variable: str, prefix: str, index: int) -> float | None:
        if variable not in metrics:
            return None
        own_depth = matched[f"depth_{variable}"]
        position = int(np.argmin(np.abs(own_depth - depths[index])))
        # Only report it if that level really is the same depth.
        if abs(float(own_depth[position]) - float(depths[index])) > 1.0:
            return None
        return round(float(matched[f"{prefix}_{variable}"][position]), 3)

    levels = [
        {
            "depth": round(float(depths[i]), 2),
            "argo_temp": at("sst", "observed", i),
            "model_temp": at("sst", "model", i),
            "argo_salinity": at("salinity", "observed", i),
            "model_salinity": at("salinity", "model", i),
        }
        for i in range(0, depths.size, step)
    ]

    # --- collocation quality ------------------------------------------------
    # Reported, never used to hide the result: a wide match is still shown, just
    # flagged so nobody reads it as clean validation.
    warnings: list[str] = []
    if time_separation_hours is not None and time_separation_hours > max_time_separation_hours:
        warnings.append(
            f"temporal separation {time_separation_hours:.1f} h exceeds "
            f"{max_time_separation_hours:.0f} h"
        )
    if distance_km is not None and distance_km > max_distance_km:
        warnings.append(
            f"spatial separation {distance_km:.1f} km exceeds {max_distance_km:.0f} km"
        )

    return {
        "sensor_id": sensor["id"],
        "cycle": sensor["cycle"],
        "lat": lat,
        "lon": lon,
        "observed_time": sensor["time"],
        "model_date": model_date,
        "model_time": model_instant.isoformat(),
        "requested_date": requested,
        "date_offset_days": date_offset_days,
        "time_separation_hours": time_separation_hours,
        "distance_km": distance_km,
        "collocation": {
            "argo_lat": lat,
            "argo_lon": lon,
            "argo_time": sensor["time"],
            "model_lat": round(grid_lat, 4) if grid_lat is not None else None,
            "model_lon": round(grid_lon, 4) if grid_lon is not None else None,
            "model_time": model_instant.isoformat(),
            "time_separation_hours": time_separation_hours,
            "distance_km": distance_km,
            "max_time_separation_hours": max_time_separation_hours,
            "max_distance_km": max_distance_km,
        },
        "confidence": "low" if warnings else "ok",
        "confidence_warnings": warnings,
        "model_grid_lat": round(grid_lat, 4) if grid_lat is not None else None,
        "model_grid_lon": round(grid_lon, 4) if grid_lon is not None else None,
        "model_depth_range": model_depth_range,
        "observed_levels": len(observed_levels),
        "matched_levels": int(depths.size),
        "plotted_levels": len(levels),
        "levels": levels,
        "metrics": metrics,
    }


def get_buoy(sensor_id: str) -> dict | None:
    for buoy in MOORED_BUOYS:
        if buoy["id"].upper() == sensor_id.upper():
            return buoy
    return None


def get_sensor_profile(sensor_id: str) -> dict:
    """Depth profile for one sensor -- real for Argo, synthetic for buoys."""
    buoy = get_buoy(sensor_id)

    if buoy is not None:
        # Buoys observe at the surface only, so the profile is generated with
        # the same decay formula main.js uses.
        profile = [
            {
                "depth": float(depth),
                "temp": subsurface_value(buoy["obsTemp"], depth, "sst"),
                "salinity": subsurface_value(buoy["obsSalinity"], depth, "salinity"),
                "wave": subsurface_value(buoy["obsWave"], depth, "wave"),
            }
            for depth in SYNTHETIC_DEPTHS
        ]
        return {
            "data_source": "live",       # static record; nothing to sync
            "last_synced_at": None,
            "id": buoy["id"],
            "name": buoy["name"],
            "place": buoy["place"],
            "lat": buoy["lat"],
            "lon": buoy["lon"],
            "source": SOURCE_PLACEHOLDER,
            "synthetic": True,
            "cycle": None,
            "time": None,
            "profile": profile,
        }

    # Anything else is treated as an Argo id.
    try:
        wmo = argo_service.normalise_float_id(sensor_id)
    except ValueError:
        raise SensorNotFoundError(
            f"unknown sensor {sensor_id!r} -- expected a buoy id like "
            f"{MOORED_BUOYS[0]['id']} or an Argo id like ARGO-1902367"
        )

    try:
        profile = argo_service.get_float_profile(wmo)
    except argo_service.FloatNotFoundError as exc:
        raise SensorNotFoundError(str(exc)) from exc

    return {
        # Provenance of the underlying Argo fetch: live GDAC or persistent cache.
        "data_source": profile.get("source", "live"),
        "last_synced_at": profile.get("last_synced_at"),
        "id": profile["id"],
        "name": f"Argo Float {profile['wmo']}",
        "place": f"{_basin_for(profile['lat'], profile['lon'])} (cycle {profile['cycle']})",
        "lat": profile["lat"],
        "lon": profile["lon"],
        "source": SOURCE_ARGO_LIVE,
        "synthetic": False,
        "cycle": profile["cycle"],
        "time": profile["time"],
        # Argo has no wave sensor, so that key stays null at every level.
        "profile": [{**level, "wave": None} for level in profile["profile"]],
    }
