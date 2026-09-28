"""
The state of each named sea region on one model day, for the globe's ticker.

For a handful of well-known regions of the model domain, the surface figures a
reader wants at a glance -- each read straight off the downloaded fields for
that day and labelled with its kind:

  sea temperature     model, top level (~0.5 m): mean and range
  sea surface salinity  model, top level: mean
  current             model, top level: strongest speed, sqrt(uo^2 + vo^2)
  wave height         model, the midday 3-hourly step: mean and highest
  mixed layer depth   model: mean
  sea level anomaly   satellite altimetry: mean, in cm
  chlorophyll-a       satellite ocean colour, gap-filled: median; the satellite
                      lags the model by about two days, so the nearest day it
                      has is used and named

plus, when Ocean Connect already holds a copy, the number of coastal areas
(districts) whose alert centre lies in the region's box and that are under an
INCOIS or SACHET alert in force NOW -- a different clock from the model day,
and said so.

Means are over the region's water cells; land is skipped, never counted as 0.
Results are cached per day and set of files.
"""

from __future__ import annotations

import threading

import numpy as np
import xarray as xr

from app.services import ocean_model
from app.services.ocean_model import DatasetNotFoundError

# (key, name, (min_lon, min_lat, max_lon, max_lat)) -- all inside the model domain.
REGIONS = [
    ("bay_of_bengal", "Bay of Bengal", (80.0, 5.0, 92.0, 22.0)),
    ("north_bay", "Northern Bay of Bengal", (85.0, 18.0, 92.5, 22.5)),
    ("andaman_sea", "Andaman Sea", (92.5, 6.0, 96.0, 16.0)),
    ("arabian_sea", "Arabian Sea", (62.0, 8.0, 75.0, 24.0)),
    ("laccadive_sea", "Laccadive Sea", (71.0, 6.0, 77.5, 13.0)),
    ("gulf_of_mannar", "Gulf of Mannar", (77.8, 8.0, 79.5, 9.6)),
]

_cache: dict[tuple, dict] = {}
_lock = threading.Lock()


def _box(da: xr.DataArray, bbox) -> xr.DataArray:
    lon0, lat0, lon1, lat1 = bbox
    lat = ocean_model._coord_name(da, "latitude", "lat")
    lon = ocean_model._coord_name(da, "longitude", "lon")
    lats = da[lat].values
    lat_slice = slice(lat0, lat1) if lats[0] <= lats[-1] else slice(lat1, lat0)
    return da.sel({lat: lat_slice, lon: slice(lon0, lon1)})


def _surface(ds: xr.Dataset, nc_var: str) -> xr.DataArray:
    da = ds[nc_var]
    if "time" in da.dims:
        da = da.isel(time=0)
    depth = ocean_model._depth_dim(da)
    if depth is not None:
        da = da.isel({depth: 0})
    return da


def _load(variable: str, day: str):
    """(dataset, day its data is really from). A file that lacks the day serves its
    nearest step, and a missing file the nearest downloaded day; either way the day
    reported is the data's own."""
    try:
        ds = ocean_model.load_dataset(variable, day)
    except DatasetNotFoundError:
        nearest = ocean_model.find_nearest_available_date(variable, day)
        if nearest is None:
            return None, None
        ds = ocean_model.load_dataset(variable, str(nearest))
    if "time" in ds.coords and ds["time"].size:
        return ds, str(np.datetime_as_string(np.atleast_1d(ds["time"].values)[0], unit="D"))
    return ds, day


def _stat(values: np.ndarray, how: str) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float({"mean": np.mean, "max": np.max, "min": np.min, "median": np.median}[how](values))


def _r(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _alerts_now() -> tuple[list[dict], str | None]:
    """Coastal areas under an alert in force now, from Ocean Connect's copy (never fetched here)."""
    try:
        from app.services import ocean_connect
        copy = ocean_connect._last_copy("events")
    except Exception:  # noqa: BLE001 -- the ticker stands without alerts
        return [], None
    if not copy:
        return [], None
    # One entry per coastal area, however many alerts it is under.
    areas: dict[tuple, dict] = {}
    for e in copy.get("events", []):
        if not e.get("active") or e.get("class") in ("EARTHQUAKE",):
            continue
        for a in (e.get("areas") or [{"lat": e.get("lat"), "lon": e.get("lon")}]):
            if a.get("lat") is not None and a.get("lon") is not None:
                area = areas.setdefault((a["lat"], a["lon"]), {"lat": a["lat"], "lon": a["lon"], "classes": set()})
                area["classes"].add(e["class"])
    return list(areas.values()), copy.get("fetched_at")


def get_summary(date_str: str) -> dict:
    variables = ("sst", "salinity", "uo", "vo", "wave", "mld", "sla", "chl")
    paths = []
    for v in variables:
        try:
            paths.append(tuple(str(p) for p in ocean_model.find_dataset_paths(v, date_str)))
        except DatasetNotFoundError:
            paths.append(())
    key = (date_str, tuple(paths))
    with _lock:
        cached = _cache.get(key)
    if cached:
        return _with_alerts(dict(cached))

    fields: dict[str, xr.DataArray] = {}
    used_day: dict[str, str | None] = {}
    for v in variables:
        ds, day = _load(v, date_str)
        used_day[v] = day
        if ds is not None:
            fields[v] = _surface(ds, ocean_model.resolve_nc_variable(v))
    speed = None
    if "uo" in fields and "vo" in fields:
        speed = np.hypot(fields["uo"], fields["vo"])

    regions = []
    for key_name, name, bbox in REGIONS:
        def vals(v):
            return _box(fields[v], bbox).values.astype("float64") if v in fields else np.array([])
        sst = vals("sst")
        swh = vals("wave")
        regions.append({
            "key": key_name, "name": name, "bbox": list(bbox),
            "sst_mean_c": _r(_stat(sst, "mean"), 2), "sst_min_c": _r(_stat(sst, "min"), 2), "sst_max_c": _r(_stat(sst, "max"), 2),
            "sss_mean_psu": _r(_stat(vals("salinity"), "mean"), 2),
            "current_max_ms": _r(_stat(_box(speed, bbox).values.astype("float64"), "max"), 2) if speed is not None else None,
            "wave_mean_m": _r(_stat(swh, "mean"), 2), "wave_max_m": _r(_stat(swh, "max"), 2),
            "mld_mean_m": _r(_stat(vals("mld"), "mean"), 1),
            "sla_mean_cm": _r(None if _stat(vals("sla"), "mean") is None else _stat(vals("sla"), "mean") * 100, 1),
            "chl_median_mg_m3": _r(_stat(vals("chl"), "median"), 2),
        })

    result = {
        "date": date_str,
        "days_used": {"chlorophyll": used_day.get("chl"), "sea_level_anomaly": used_day.get("sla"),
                      "waves": used_day.get("wave")},
        "regions": regions,
        "kinds": {"sea temperature, salinity, current, waves, mixed layer": "model",
                  "sea level anomaly, chlorophyll": "satellite"},
        "note": ("Surface figures for the model day, over each region's water cells. Waves are the model's midday step; "
                 "chlorophyll is gap-filled satellite data from the nearest day it covers. Not forecasts or warnings."),
    }
    with _lock:
        _cache[key] = result
        while len(_cache) > 14:
            _cache.pop(next(iter(_cache)))
    return _with_alerts(dict(result))


def _with_alerts(result: dict) -> dict:
    points, fetched = _alerts_now()
    regions = []
    for r in result["regions"]:
        lon0, lat0, lon1, lat1 = r["bbox"]
        inside = [p for p in points if lat0 <= p["lat"] <= lat1 and lon0 <= p["lon"] <= lon1]
        regions.append({**r, "alerts_now": len(inside) if fetched else None,
                        "alert_classes": sorted({c for p in inside for c in p["classes"]})})
    return {**result, "regions": regions, "alerts_fetched_at": fetched,
            "alerts_note": "Coastal areas under an INCOIS or SACHET alert in force now (a different clock from the model day)."}
