"""
Full-depth water columns at four Indian Ocean basins, surface to seafloor.

The downloaded model files stop at 2000 m, so a descent to the seafloor needs
more. For each basin's point this module fetches the model's WHOLE column for
one day from Copernicus Marine and keeps it on disk:

  temperature  cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m  (thetao, deg C)
  salinity     cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m      (so, PSU)
  oxygen       cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m      (o2, mmol/m3)

A first request for a basin and day downloads three single-cell columns (about
half a minute, all three at once); every later one reads the cached files.

What else is served, and what kind of value each is:

  seafloor     GEBCO 2020 bathymetry at the point, via the Open Topo Data API
               (cached). The model's own seafloor is its deepest wet level,
               which is coarser; both are reported.
  pressure     DERIVED from depth and latitude, Saunders (1981).
  sunlight     ESTIMATED: the fraction of surface blue-green (490 nm) light left
               at each depth, exp(-Kd z), with Kd(490) from the day's satellite
               chlorophyll by Morel & Maritorena (2001). Not a measurement.
  Argo QC      the nearest live Argo float's latest measured cast against the
               model at the FLOAT's position (app/services/sensors.compare_profile),
               level by level, with a threshold on the temperature difference.

Nothing is filled in below the model's deepest level or the float's deepest
measurement: those depths come back null.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from app.services import ocean_model
from app.services import sensors as sensor_service
from app.services.netcdf_lock import NETCDF_LOCK

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
COLUMN_DIR = BACKEND_DIR.parent / "data" / "model" / "columns"
GEBCO_CACHE = BACKEND_DIR.parent / "data" / "observations" / "gebco_cache" / "basins.json"
GEBCO_URL = "https://api.opentopodata.org/v1/gebco2020"

# One representative deep point per basin, in chip order.
BASINS: dict[str, dict] = {
    "bay_of_bengal": {"name": "Bay of Bengal", "region": "Bay of Bengal · central basin", "lat": 8.0, "lon": 88.0},
    "arabian_sea": {"name": "Arabian Sea", "region": "Arabian Sea · Arabian Basin", "lat": 12.0, "lon": 65.0},
    "andaman_sea": {"name": "Andaman Sea", "region": "Andaman Sea · Andaman Basin", "lat": 10.5, "lon": 95.5},
    "laccadive_sea": {"name": "Laccadive Sea", "region": "Laccadive Sea · off south-west India", "lat": 8.0, "lon": 74.5},
}

PRODUCTS = {
    "temperature": ("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", "thetao"),
    "salinity": ("cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", "so"),
    "oxygen": ("cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m", "o2"),
}
BOX_HALF_WIDTH = 0.15            # deg around the point, so a cell is always inside
O2_MMOL_PER_ML = 44.661          # 1 ml/L of O2 = 44.661 umol/L (mmol/m3)
ARGO_RADIUS_KM = 600.0           # nearest float searched within this distance
QC_THRESHOLD_C = 0.6             # |Argo - model| at or above this: drift (amber)
QC_FAIL_C = 1.2                  # ... at or above this: failing (red)

SOURCES = {
    "temperature": {"kind": "model", "dataset": PRODUCTS["temperature"][0]},
    "salinity": {"kind": "model", "dataset": PRODUCTS["salinity"][0]},
    "oxygen": {"kind": "model", "dataset": PRODUCTS["oxygen"][0], "note": "converted to ml/L (1 ml/L = 44.661 mmol/m3)"},
    "pressure": {"kind": "derived", "method": "Saunders (1981), from depth and latitude"},
    "sunlight": {"kind": "estimate", "method": "exp(-Kd(490) z), Kd(490) from satellite chlorophyll, Morel & Maritorena (2001)"},
    "seafloor": {"kind": "bathymetry", "dataset": "GEBCO 2020 via Open Topo Data"},
    "argo": {"kind": "in-situ", "method": "nearest live Argo float's latest cast vs the model at the float"},
}


class WaterColumnError(RuntimeError):
    pass


class UnknownBasinError(KeyError):
    pass


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


# --- model columns -------------------------------------------------------------

def _download(dataset_id: str, variable: str, lat: float, lon: float, date: str, target: Path) -> None:
    """Fetch one day of one variable, all depths, in a small box. Kept separate for tests."""
    import copernicusmarine

    target.parent.mkdir(parents=True, exist_ok=True)
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_longitude=lon - BOX_HALF_WIDTH, maximum_longitude=lon + BOX_HALF_WIDTH,
        minimum_latitude=lat - BOX_HALF_WIDTH, maximum_latitude=lat + BOX_HALF_WIDTH,
        start_datetime=f"{date}T00:00:00", end_datetime=f"{date}T23:59:59",
        minimum_depth=0, maximum_depth=5728,   # deepest level of both products
        output_directory=str(target.parent), output_filename=target.name, overwrite=True,
        username=os.getenv("COPERNICUSMARINE_SERVICE_USERNAME"),
        password=os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD"),
        disable_progress_bar=True,
    )


def _column_file(basin: str, kind: str, date: str) -> Path:
    return COLUMN_DIR / f"{basin}_{kind}_{date}.nc"


def _ensure_columns(basin: str, date: str) -> dict[str, Path]:
    site = BASINS[basin]
    paths = {kind: _column_file(basin, kind, date) for kind in PRODUCTS}
    with _lock_for(f"{basin}:{date}"):
        missing = [kind for kind, path in paths.items() if not path.exists()]
        if missing:
            logger.info("water column %s %s: downloading %s", basin, date, ", ".join(missing))
            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures = {
                    kind: pool.submit(_download, *PRODUCTS[kind], site["lat"], site["lon"], date, paths[kind])
                    for kind in missing
                }
                errors = []
                for kind, future in futures.items():
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 -- reported, never papered over
                        errors.append(f"{kind}: {exc}")
                        paths[kind].unlink(missing_ok=True)
            if errors:
                raise WaterColumnError("Copernicus Marine download failed -- " + "; ".join(errors))
    return paths


def _nearest_wet_column(path: Path, variable: str, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(depths, values, cell_lat, cell_lon) of the wet cell nearest the point."""
    with NETCDF_LOCK, xr.open_dataset(path) as ds:
        da = ds[variable]
        if "time" in da.dims:
            da = da.isel(time=0)
        depths = da["depth"].values.astype(float)
        best = None
        for la in da["latitude"].values:
            for lo in da["longitude"].values:
                values = da.sel(latitude=la, longitude=lo).values.astype(float)
                if not np.isfinite(values[0]):
                    continue
                d = ocean_model.haversine_km(lat, lon, float(la), float(lo))
                if best is None or d < best[0]:
                    best = (d, values, float(la), float(lo))
    if best is None:
        raise WaterColumnError(f"no wet model cell within {BOX_HALF_WIDTH} deg of {lat}, {lon} in {path.name}")
    return depths, best[1], best[2], best[3]


# --- derived quantities ------------------------------------------------------------

def pressure_mpa(depth_m: float, lat: float) -> float:
    """Sea pressure from depth, Saunders (1981): p in dbar, returned in MPa."""
    c1 = (5.92 + 5.25 * math.sin(math.radians(lat)) ** 2) * 1e-3
    dbar = ((1 - c1) - math.sqrt((1 - c1) ** 2 - 8.84e-6 * depth_m)) / 4.42e-6
    return dbar / 100.0


def kd490(chl: float) -> float:
    """Diffuse attenuation of 490 nm light from chlorophyll, Morel & Maritorena (2001)."""
    return 0.0166 + 0.07242 * chl ** 0.68955


def _surface_chlorophyll(lat: float, lon: float, date: str) -> tuple[float | None, str | None]:
    box = (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)
    try:
        points, _ = ocean_model.get_grid("chl", date, 0, bbox=box)
    except Exception:  # noqa: BLE001 -- no chlorophyll that day: sunlight is simply not estimated
        return None, None
    if not points:
        return None, None
    nearest = min(points, key=lambda p: (p["lat"] - lat) ** 2 + (p["lon"] - lon) ** 2)
    return float(nearest["value"]), ocean_model.provenance("chl")[1]


# --- seafloor -----------------------------------------------------------------------

def _fetch_gebco() -> dict[str, float]:
    locations = "|".join(f"{b['lat']},{b['lon']}" for b in BASINS.values())
    url = f"{GEBCO_URL}?{urllib.parse.urlencode({'locations': locations})}"
    request = urllib.request.Request(url, headers={"User-Agent": "OCEANAO/0.1 (SIH ocean platform)"})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.load(response)
    results = body.get("results") or []
    if len(results) != len(BASINS):
        raise WaterColumnError("GEBCO lookup returned the wrong number of points")
    return {key: -float(r["elevation"]) for key, r in zip(BASINS, results)}


def seafloor_depths() -> dict[str, float | None]:
    """GEBCO depth at each basin point, cached on disk once fetched."""
    if GEBCO_CACHE.exists():
        try:
            cached = json.loads(GEBCO_CACHE.read_text())
            if set(cached) == set(BASINS):
                return cached
        except ValueError:
            pass
    try:
        depths = _fetch_gebco()
    except Exception as exc:  # noqa: BLE001
        logger.warning("GEBCO bathymetry unavailable: %s", exc)
        return {key: None for key in BASINS}
    GEBCO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GEBCO_CACHE.write_text(json.dumps(depths))
    return depths


RELIEF_CACHE = GEBCO_CACHE.parent / "relief.json"
RELIEF_N = 10                    # 10 x 10 GEBCO points (the API's limit is 100 per call)
RELIEF_HALF_DEG = 1.0            # around the basin point


def _fetch_relief(lat: float, lon: float) -> dict:
    lats = [lat - RELIEF_HALF_DEG + 2 * RELIEF_HALF_DEG * j / (RELIEF_N - 1) for j in range(RELIEF_N)]
    lons = [lon - RELIEF_HALF_DEG + 2 * RELIEF_HALF_DEG * i / (RELIEF_N - 1) for i in range(RELIEF_N)]
    locations = "|".join(f"{la:.4f},{lo:.4f}" for la in lats for lo in lons)
    url = f"{GEBCO_URL}?{urllib.parse.urlencode({'locations': locations})}"
    request = urllib.request.Request(url, headers={"User-Agent": "OCEANAO/0.1 (SIH ocean platform)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        results = json.load(response).get("results") or []
    if len(results) != RELIEF_N * RELIEF_N:
        raise WaterColumnError("GEBCO relief lookup returned the wrong number of points")
    depths = [-float(r["elevation"]) for r in results]
    return {"lats": [round(v, 4) for v in lats], "lons": [round(v, 4) for v in lons],
            "depths": [depths[j * RELIEF_N:(j + 1) * RELIEF_N] for j in range(RELIEF_N)]}


def seafloor_relief(basin: str) -> dict | None:
    """GEBCO depths on a 10 x 10 grid +/-1 deg around the basin point, cached."""
    cache = {}
    if RELIEF_CACHE.exists():
        try:
            cache = json.loads(RELIEF_CACHE.read_text())
        except ValueError:
            cache = {}
    if basin in cache:
        return cache[basin]
    site = BASINS[basin]
    try:
        relief = _fetch_relief(site["lat"], site["lon"])
    except Exception as exc:  # noqa: BLE001 -- no relief: the floor is drawn flat, and says so
        logger.warning("GEBCO relief for %s unavailable: %s", basin, exc)
        return None
    cache[basin] = relief
    RELIEF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RELIEF_CACHE.write_text(json.dumps(cache))
    return relief


# --- Argo QC ------------------------------------------------------------------------

def _argo_qc(lat: float, lon: float) -> dict:
    try:
        floats = [s for s in sensor_service.list_sensors("full_domain", include_argo=True, include_buoys=False)
                  if s.get("source") == "argo_live"]
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"Argo floats unavailable ({exc})"}
    if not floats:
        return {"available": False, "reason": "no live Argo floats in the domain"}
    nearest = min(floats, key=lambda s: ocean_model.haversine_km(lat, lon, s["lat"], s["lon"]))
    distance = ocean_model.haversine_km(lat, lon, nearest["lat"], nearest["lon"])
    if distance > ARGO_RADIUS_KM:
        return {"available": False, "reason": f"no live Argo float within {ARGO_RADIUS_KM:.0f} km"}
    try:
        cmp = sensor_service.compare_profile(nearest["id"], None, max_levels=400)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"comparison with {nearest['id']} failed ({exc})"}
    levels = [
        {"depth": lv["depth"], "argo_temp": lv["argo_temp"], "model_temp": lv["model_temp"]}
        for lv in cmp["levels"] if lv["argo_temp"] is not None and lv["model_temp"] is not None
    ]
    return {
        "available": bool(levels),
        "reason": None if levels else f"{nearest['id']} has no depths matched with the model",
        "float_id": cmp["sensor_id"],
        "cycle": cmp.get("cycle"),
        "float_lat": cmp["lat"],
        "float_lon": cmp["lon"],
        "distance_km": round(distance, 1),
        "observed_time": cmp.get("observed_time"),
        "model_date": cmp["model_date"],
        "time_separation_hours": cmp.get("time_separation_hours"),
        "confidence": cmp["confidence"],
        "confidence_warnings": cmp.get("confidence_warnings", []),
        "threshold_c": QC_THRESHOLD_C,
        "fail_c": QC_FAIL_C,
        "levels": levels,
    }


# --- the column ---------------------------------------------------------------------

COLUMN_CACHE_SECONDS = 3600      # the Argo comparison inside takes seconds; the day's column does not change
_column_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def get_column(basin: str, date: str) -> dict:
    if basin not in BASINS:
        raise UnknownBasinError(basin)
    import time
    cached = _column_cache.get((basin, date))
    if cached and time.monotonic() - cached[0] < COLUMN_CACHE_SECONDS:
        return cached[1]
    result = _build_column(basin, date)
    _column_cache[(basin, date)] = (time.monotonic(), result)
    return result


def _build_column(basin: str, date: str) -> dict:
    site = BASINS[basin]
    paths = _ensure_columns(basin, date)

    t_depths, temps, cell_lat, cell_lon = _nearest_wet_column(paths["temperature"], "thetao", site["lat"], site["lon"])
    s_depths, sals, _, _ = _nearest_wet_column(paths["salinity"], "so", site["lat"], site["lon"])
    o_depths, oxy, o_lat, o_lon = _nearest_wet_column(paths["oxygen"], "o2", site["lat"], site["lon"])

    wet = np.isfinite(temps)
    model_deepest = float(t_depths[wet][-1])
    chl, chl_dataset = _surface_chlorophyll(site["lat"], site["lon"], date)
    kd = kd490(chl) if chl is not None else None
    o_wet = np.isfinite(oxy)

    levels = []
    for i, depth in enumerate(t_depths):
        if not wet[i]:
            break
        sal = float(sals[np.argmin(np.abs(s_depths - depth))])
        # Oxygen lives on its own grid: interpolate in depth, never beyond its deepest wet level.
        o2 = None
        if o_wet.any() and o_depths[o_wet][0] <= depth <= o_depths[o_wet][-1]:
            o2 = float(np.interp(depth, o_depths[o_wet], oxy[o_wet])) / O2_MMOL_PER_ML
        elif o_wet.any() and depth < o_depths[o_wet][0]:
            o2 = float(oxy[o_wet][0]) / O2_MMOL_PER_ML
        levels.append({
            "depth": round(float(depth), 3),
            "temperature": round(float(temps[i]), 4),
            "salinity": round(sal, 4) if math.isfinite(sal) else None,
            "oxygen_ml_l": round(o2, 3) if o2 is not None else None,
            "pressure_mpa": round(pressure_mpa(float(depth), site["lat"]), 3),
            "sunlight_fraction": (math.exp(-kd * float(depth)) if kd is not None else None),
        })

    seafloor = seafloor_depths().get(basin)
    return {
        "basin": basin,
        "name": site["name"],
        "region": site["region"],
        "lat": site["lat"],
        "lon": site["lon"],
        "date": date,
        "model_cell": {"lat": round(cell_lat, 4), "lon": round(cell_lon, 4)},
        "oxygen_cell": {"lat": round(o_lat, 4), "lon": round(o_lon, 4)},
        "seafloor_m": seafloor,
        "relief": seafloor_relief(basin),
        "model_deepest_m": round(model_deepest, 3),
        "chlorophyll_mg_m3": round(chl, 4) if chl is not None else None,
        "chlorophyll_dataset": chl_dataset,
        "kd490": round(kd, 5) if kd is not None else None,
        "levels": levels,
        "argo": _argo_qc(site["lat"], site["lon"]),
        "sources": SOURCES,
    }


def list_basins() -> list[dict]:
    depths = seafloor_depths()
    return [{"basin": key, **value, "seafloor_m": depths.get(key)} for key, value in BASINS.items()]
