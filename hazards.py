"""
Hazard outlooks: the next 72 hours of five ocean hazards over the model domain.

Each hazard is read from a real forecast and turned into a map per lead time by
a stated rule. Nothing is scored, blended or learned:

  cyclones & gales  INCOIS Ocean State Forecast (WAVEWATCH III): 10 m wind and
                    significant wave height, 3-hourly. Gale-force wind is 34 kt
                    or more, storm-force 48 kt, hurricane-force 64 kt (Beaufort /
                    WMO); high seas are waves of 4 m or more. Active storms come
                    from IBTrACS separately (the Cyclone layer).
  marine heatwaves  Copernicus global analysis-forecast SST (daily means, top
                    level) against the Hobday et al. (2016) threshold for each
                    cell and day of year -- the same 1993-2022 GLORYS baseline the
                    heatwave record uses. A heatwave needs five days above the
                    threshold; a 4-day forecast can only show days above it, so
                    that is what it says.
  low oxygen        Copernicus BGC analysis-forecast dissolved oxygen, 0-400 m:
                    the shallowest depth where O2 falls below 60 mmol/m3 (about
                    2 mg/L, the usual hypoxia line; Vaquer-Sunyer & Duarte 2008).
                    A shallow hypoxic boundary squeezes fish into a thin surface
                    layer.
  algal blooms      Copernicus BGC analysis-forecast chlorophyll (model), and the
                    latest satellite chlorophyll (observed, L4 gap-filled).
                    Chlorophyll of 5 mg/m3 or more is taken as bloom-level, 10 as
                    intense. High chlorophyll says a bloom is likely; it cannot
                    say whether the species is harmful -- that needs sampling.
  oil spill drift   a Lagrangian drift for a hypothetical release: particles moved
                    by Copernicus 6-hourly forecast surface currents plus 3% of
                    the INCOIS forecast wind, with a random-walk spread. No live
                    oil-spill feed exists for these seas; this shows where oil
                    released at a point now would drift, not how much would remain.

Forecast files are fetched from Copernicus Marine once per issue day and kept in
data/model/hazards/; INCOIS fields are read over OPeNDAP (see forecast.py).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from app.services import forecast as incois
from app.services import heatwave_store
from app.services.netcdf_lock import NETCDF_LOCK

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
HAZARD_DIR = BACKEND_DIR.parent / "data" / "model" / "hazards"
DOMAIN = (62.0, 4.0, 96.0, 24.0)                 # lon min, lat min, lon max, lat max
LEADS_H = [0, 24, 48, 72]
KM_PER_DEG = 111.32

GALE_KT, STORM_KT, HURRICANE_KT = 34.0, 48.0, 64.0
HIGH_SEAS_M = 4.0
HYPOXIA_MMOL_M3 = 60.0
SHALLOW_HYPOXIA_M = 100.0
BLOOM_MG_M3, INTENSE_BLOOM_MG_M3 = 5.0, 10.0
WINDAGE = 0.03
DIFFUSIVITY_M2_S = 10.0

PRODUCTS = {
    "sst": ("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ["thetao"], (0.0, 1.0)),
    "o2": ("cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m", ["o2"], (0.0, 400.0)),
    "chl": ("cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m", ["chl"], (0.0, 1.0)),
    "cur": ("cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i", ["uo", "vo"], (0.0, 1.0)),
    "satchl": ("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", ["CHL"], None),
}

HAZARDS = ("cyclones", "heatwaves", "low_oxygen", "algal_blooms")


class HazardError(RuntimeError):
    pass


class UnknownHazardError(KeyError):
    pass


_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()
_outlooks: dict[tuple[str, str], dict] = {}


def _lock(key: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(key, threading.Lock())


def issue_day(now: datetime | None = None) -> date:
    return (now or datetime.now(timezone.utc)).date()


# --- Copernicus forecast files ----------------------------------------------------

def _download(product: str, start: date, end: date, target: Path) -> None:
    """Fetch one product over the domain for a date range. Kept separate for tests."""
    import copernicusmarine

    dataset, variables, depths = PRODUCTS[product]
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        dataset_id=dataset, variables=variables,
        minimum_longitude=DOMAIN[0], maximum_longitude=DOMAIN[2],
        minimum_latitude=DOMAIN[1], maximum_latitude=DOMAIN[3],
        start_datetime=f"{start}T00:00:00", end_datetime=f"{end}T23:59:59",
        output_directory=str(target.parent), output_filename=target.name, overwrite=True,
        username=os.getenv("COPERNICUSMARINE_SERVICE_USERNAME"),
        password=os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD"),
        disable_progress_bar=True,
    )
    if depths:
        kwargs.update(minimum_depth=depths[0], maximum_depth=depths[1])
    copernicusmarine.subset(**kwargs)


def product_file(product: str, day: date) -> Path:
    """The product's file for an issue day, downloaded on first use."""
    target = HAZARD_DIR / f"{product}_{day.isoformat()}.nc"
    with _lock(f"{product}:{day}"):
        if not target.exists():
            if product == "satchl":
                # The satellite product lags a few days: take the last week, use the newest.
                start, end = day - timedelta(days=7), day
            else:
                start, end = day, day + timedelta(days=3)
            try:
                _download(product, start, end, target)
            except Exception as exc:  # noqa: BLE001
                target.unlink(missing_ok=True)
                raise HazardError(f"Copernicus Marine download of {PRODUCTS[product][0]} failed: {exc}") from exc
    return target


# --- grids --------------------------------------------------------------------------

def _grid(lats: np.ndarray, lons: np.ndarray, stride: int = 1) -> dict:
    lats, lons = lats[::stride], lons[::stride]
    return {"lat0": round(float(lats[0]), 4), "lon0": round(float(lons[0]), 4),
            "dlat": round(float(lats[1] - lats[0]), 5), "dlon": round(float(lons[1] - lons[0]), 5),
            "ny": int(lats.size), "nx": int(lons.size)}


def _flat(values: np.ndarray, digits: int) -> list:
    """Row-major (south first) list, None over land."""
    out = np.round(values.astype("float64"), digits).ravel().tolist()
    return [None if (v is None or not math.isfinite(v)) else v for v in out]


def _cell_area_km2(lats: np.ndarray, dlat: float, dlon: float) -> np.ndarray:
    return (KM_PER_DEG * abs(dlat)) * (KM_PER_DEG * abs(dlon) * np.cos(np.radians(lats)))


def _area(mask: np.ndarray, lats: np.ndarray, dlat: float, dlon: float) -> float:
    areas = _cell_area_km2(lats, dlat, dlon)[:, None] * np.ones(mask.shape[1])[None, :]
    return round(float(np.nansum(np.where(mask, areas, 0.0))), 0)


def _daily_frames(ds: xr.Dataset, day: date) -> list[int]:
    """Indices of the four daily steps: the issue day and the next three."""
    dates = np.asarray(ds["time"].values, dtype="datetime64[D]")
    idx = []
    for k in range(4):
        hits = np.flatnonzero(dates == np.datetime64(day + timedelta(days=k), "D"))
        if hits.size == 0:
            raise HazardError(f"the forecast file has no step for {day + timedelta(days=k)}")
        idx.append(int(hits[0]))
    return idx


# --- the four gridded outlooks ------------------------------------------------------------

def _outlook_heatwaves(day: date) -> dict:
    with NETCDF_LOCK, xr.open_dataset(product_file("sst", day)) as ds:
        idx = _daily_frames(ds, day)
        sst = ds["thetao"].isel(depth=0).values[idx].astype("float64")
        lats, lons = ds["latitude"].values, ds["longitude"].values
        times = [str(np.datetime_as_string(ds["time"].values[i], unit="m")) + "Z" for i in idx]
    mean, p90, clat, clon, attrs = heatwave_store._open_climatology(str(heatwave_store.data_dir()))
    if clat.size != lats.size or clon.size != lons.size or np.abs(clat - lats).max() > 1e-3 or np.abs(clon - lons).max() > 1e-3:
        raise HazardError("the forecast grid does not match the heatwave climatology grid")
    stride = 2
    frames, summaries = [], []
    days_above = np.zeros(sst.shape[1:], dtype="int16")
    for k in range(4):
        d = day + timedelta(days=k)
        doy = min(d.timetuple().tm_yday, 365) - 1        # day 366 pooled with 365, as in the record
        m, t = mean[doy].astype("float64"), p90[doy].astype("float64")
        spread = np.where(t - m > 0.05, t - m, np.nan)
        multiple = np.where(np.isfinite(sst[k]), (sst[k] - m) / spread, np.nan)   # 1 = at the threshold
        above = multiple >= 1.0
        days_above += np.where(above, 1, 0).astype("int16")
        cats = {}
        for lo, hi, name in ((1, 2, "moderate"), (2, 3, "strong"), (3, 4, "severe"), (4, 1e9, "extreme")):
            cats[name] = _area((multiple >= lo) & (multiple < hi), lats, lats[1] - lats[0], lons[1] - lons[0])
        frames.append(_flat(multiple[::stride, ::stride], 2))
        summaries.append({
            "area_km2": _area(above, lats, lats[1] - lats[0], lons[1] - lons[0]),
            "by_category_km2": cats,
            "max_multiple": round(float(np.nanmax(multiple)), 2),
            "max_excess_c": round(float(np.nanmax(sst[k] - t)), 2),
        })
    return {
        "hazard": "heatwaves",
        "title": "Marine heatwave conditions",
        "value": "multiple of the threshold excess (sea temperature − mean) ÷ (threshold − mean); 1 = at the 90th-percentile threshold",
        "units": "×",
        "classes": [{"from": 1, "label": "Above threshold (moderate, cat. I)", "colour": "#fbbf24"},
                    {"from": 2, "label": "Strong (II)", "colour": "#f97316"},
                    {"from": 3, "label": "Severe (III)", "colour": "#ef4444"},
                    {"from": 4, "label": "Extreme (IV)", "colour": "#7f1d1d"}],
        "scale": {"min": -1, "max": 4},
        "times": times, "time_kind": "daily mean",
        "grid": _grid(lats, lons, stride), "frames": frames, "summaries": summaries,
        "extra": {"days_above_threshold": _flat(days_above[::stride, ::stride].astype("float64"), 0)},
        "summary_label": "Area above the heatwave threshold",
        "sources": [PRODUCTS["sst"][0], f"threshold: {attrs.get('dataset_id', 'GLORYS12V1')} {attrs.get('baseline_first_year')}–{attrs.get('baseline_last_year')}, 90th percentile, 11-day window"],
        "note": ("A marine heatwave needs five or more days above the threshold (Hobday et al. 2016); this 4-day forecast shows days above it, "
                 "not declared heatwaves. The forecast and the baseline are both Mercator models (analysis-forecast vs reanalysis). "
                 "Over 10–23 June 2026, days both cover, the forecast system averaged 0.11 °C cooler than the reanalysis across this "
                 "domain, so it is more likely to understate days above the threshold than to overstate them."),
    }


def _outlook_low_oxygen(day: date) -> dict:
    with NETCDF_LOCK, xr.open_dataset(product_file("o2", day)) as ds:
        idx = _daily_frames(ds, day)
        o2 = ds["o2"].values[idx].astype("float64")          # time, depth, lat, lon
        depths = ds["depth"].values.astype("float64")
        lats, lons = ds["latitude"].values, ds["longitude"].values
        times = [str(np.datetime_as_string(ds["time"].values[i], unit="m")) + "Z" for i in idx]
    frames, summaries = [], []
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    for k in range(4):
        col = o2[k]
        wet = np.isfinite(col[0])
        hyp = col < HYPOXIA_MMOL_M3
        # Shallowest depth where O2 drops below the line, interpolated between levels.
        boundary = np.full(col.shape[1:], np.nan)
        first = np.argmax(hyp, axis=0)
        has = hyp.any(axis=0) & wet
        for j, i in zip(*np.nonzero(has)):
            n = first[j, i]
            if n == 0:
                boundary[j, i] = depths[0]
                continue
            a, b = col[n - 1, j, i], col[n, j, i]
            f = (a - HYPOXIA_MMOL_M3) / (a - b) if a != b else 0.0
            boundary[j, i] = depths[n - 1] + f * (depths[n] - depths[n - 1])
        frames.append(_flat(boundary, 0))
        shallow = has & (boundary <= SHALLOW_HYPOXIA_M)
        summaries.append({
            "area_km2": _area(shallow, lats, dlat, dlon),
            "hypoxic_above_400m_km2": _area(has, lats, dlat, dlon),
            "shallowest_m": round(float(np.nanmin(boundary)), 1) if has.any() else None,
            "min_o2_100m": round(float(np.nanmin(col[np.argmin(np.abs(depths - 100))])), 1),
        })
    return {
        "hazard": "low_oxygen",
        "title": "Low-oxygen (hypoxic) water",
        "value": f"depth of the hypoxic boundary: the shallowest depth where dissolved oxygen falls below {HYPOXIA_MMOL_M3:.0f} mmol/m³ (≈ 2 mg/L)",
        "units": "m",
        "classes": [{"to": 50, "label": "Hypoxia within 50 m", "colour": "#7f1d1d"},
                    {"to": 100, "label": "within 100 m", "colour": "#ef4444"},
                    {"to": 200, "label": "within 200 m", "colour": "#f97316"},
                    {"to": 400, "label": "within 400 m", "colour": "#fbbf24"}],
        "scale": {"min": 0, "max": 400},
        "times": times, "time_kind": "daily mean",
        "grid": _grid(lats, lons), "frames": frames, "summaries": summaries,
        "summary_label": f"Area with hypoxia within {SHALLOW_HYPOXIA_M:.0f} m of the surface",
        "sources": [PRODUCTS["o2"][0]],
        "note": ("Hypoxia line: 60 mmol/m³ (≈ 2 mg/L), the threshold commonly used for marine hypoxia (Vaquer-Sunyer & Duarte 2008). "
                 "Blank cells have no hypoxic water in the top 400 m. The Arabian Sea and Bay of Bengal hold one of the world's "
                 "largest oxygen minimum zones; a shallow boundary is where it reaches up toward fishing depths."),
    }


def _outlook_algal_blooms(day: date) -> dict:
    with NETCDF_LOCK, xr.open_dataset(product_file("chl", day)) as ds:
        idx = _daily_frames(ds, day)
        chl = ds["chl"].isel(depth=0).values[idx].astype("float64")
        lats, lons = ds["latitude"].values, ds["longitude"].values
        times = [str(np.datetime_as_string(ds["time"].values[i], unit="m")) + "Z" for i in idx]
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    frames, summaries = [], []
    for k in range(4):
        frames.append(_flat(chl[k], 3))
        summaries.append({
            "area_km2": _area(chl[k] >= BLOOM_MG_M3, lats, dlat, dlon),
            "intense_km2": _area(chl[k] >= INTENSE_BLOOM_MG_M3, lats, dlat, dlon),
            "max_mg_m3": round(float(np.nanmax(chl[k])), 2),
        })
    # The latest satellite day, as the observed "now".
    observed = None
    try:
        with NETCDF_LOCK, xr.open_dataset(product_file("satchl", day)) as ds:
            last = int(np.flatnonzero(np.isfinite(ds["CHL"].values).any(axis=(1, 2)))[-1])
            sat = ds["CHL"].values[last].astype("float64")
            slat, slon = ds["latitude"].values, ds["longitude"].values
            if slat[0] > slat[-1]:                          # satellite grids can run north to south
                sat, slat = sat[::-1], slat[::-1]
            s = 3                                             # 4 km -> about 12 km for the map
            sdlat, sdlon = slat[1] - slat[0], slon[1] - slon[0]
            observed = {
                "time": str(np.datetime_as_string(ds["time"].values[last], unit="D")),
                "grid": _grid(slat, slon, s), "frame": _flat(sat[::s, ::s], 3),
                "summary": {"area_km2": _area(sat >= BLOOM_MG_M3, slat, sdlat, sdlon),
                            "intense_km2": _area(sat >= INTENSE_BLOOM_MG_M3, slat, sdlat, sdlon),
                            "max_mg_m3": round(float(np.nanmax(sat)), 2)},
                "source": PRODUCTS["satchl"][0],
            }
    except (HazardError, IndexError) as exc:
        logger.warning("satellite chlorophyll unavailable: %s", exc)
    return {
        "hazard": "algal_blooms",
        "title": "Algal bloom regions",
        "value": "surface chlorophyll-a",
        "units": "mg/m³",
        "classes": [{"from": BLOOM_MG_M3, "label": "Bloom-level (≥ 5 mg/m³)", "colour": "#22c55e"},
                    {"from": INTENSE_BLOOM_MG_M3, "label": "Intense (≥ 10 mg/m³)", "colour": "#a3e635"}],
        "scale": {"min": 0.05, "max": 20, "log": True},
        "times": times, "time_kind": "daily mean",
        "grid": _grid(lats, lons), "frames": frames, "summaries": summaries,
        "observed": observed,
        "summary_label": "Area with bloom-level chlorophyll (model)",
        "sources": [PRODUCTS["chl"][0]] + ([PRODUCTS["satchl"][0]] if observed else []),
        "note": ("Chlorophyll of 5 mg/m³ or more is taken as bloom-level. High chlorophyll says a bloom is likely, not whether it is "
                 "harmful: identifying the species needs water samples (INCOIS runs an Algal Bloom Information Service). "
                 "The forecast is model chlorophyll, which is smoother than the satellite's; the observed map is the latest satellite day. "
                 "In turbid coastal water, sediment can inflate satellite chlorophyll, so very high coastal values need care."),
    }


def _outlook_cyclones(day: date) -> dict:
    now = datetime.now(timezone.utc)
    run = incois._run("wave", now.date())
    x, y = np.asarray(run["x"]), np.asarray(run["y"])
    i0, i1 = int(np.abs(x - DOMAIN[0]).argmin()), int(np.abs(x - DOMAIN[2]).argmin())
    j0, j1 = int(np.abs(y - DOMAIN[1]).argmin()), int(np.abs(y - DOMAIN[3]).argmin())
    a, _ = incois._window(run["times"], now)
    steps = [a + h // 3 for h in LEADS_H]
    if steps[-1] >= len(run["times"]):
        raise HazardError("the INCOIS wave run does not reach 72 h ahead")
    s = 3
    q = f"[{steps[0]}:8:{steps[-1]}][{j0}:{s}:{j1}][{i0}:{s}:{i1}]"
    data = incois._ascii(run["path"], f"HS{q},UWND{q},VWND{q}")
    ny, nx = len(range(j0, j1 + 1, s)), len(range(i0, i1 + 1, s))
    lats, lons = y[j0:j1 + 1:s], x[i0:i1 + 1:s]

    def cube(name):
        arr = np.asarray(data[name], dtype="float64").reshape(len(steps), ny, nx)
        return np.where(arr > -900, arr, np.nan)

    hs, u, v = cube("HS"), cube("UWND"), cube("VWND")
    wind = np.hypot(u, v) * incois.MS_TO_KNOTS
    frames, summaries = [], []
    for k in range(len(steps)):
        frames.append(_flat(wind[k], 1))
        summaries.append({
            "area_km2": _area(wind[k] >= GALE_KT, lats, lats[1] - lats[0], lons[1] - lons[0]),
            "storm_km2": _area(wind[k] >= STORM_KT, lats, lats[1] - lats[0], lons[1] - lons[0]),
            "high_seas_km2": _area(hs[k] >= HIGH_SEAS_M, lats, lats[1] - lats[0], lons[1] - lons[0]),
            "max_wind_kt": round(float(np.nanmax(wind[k])), 1),
            "max_hs_m": round(float(np.nanmax(hs[k])), 2),
        })
    return {
        "hazard": "cyclones",
        "title": "Cyclones, gales and high seas",
        "value": "10 m wind speed",
        "units": "kt",
        "classes": [{"from": GALE_KT, "label": "Gale (≥ 34 kt)", "colour": "#fbbf24"},
                    {"from": STORM_KT, "label": "Storm (≥ 48 kt)", "colour": "#f97316"},
                    {"from": HURRICANE_KT, "label": "Hurricane-force (≥ 64 kt)", "colour": "#ef4444"}],
        "scale": {"min": 0, "max": 64},
        "times": [incois._iso(run["times"][k]) for k in steps], "time_kind": "instant (3-hourly model step)",
        "grid": _grid(lats, lons), "frames": frames, "summaries": summaries,
        "extra": {"wave_height_m": [_flat(hs[k], 2) for k in range(len(steps))]},
        "summary_label": "Area with gale-force wind",
        "sources": [f"INCOIS Ocean State Forecast, {run['path']}"],
        "note": ("Wind classes follow the Beaufort / WMO scale; high seas are waves of 4 m or more. These are INCOIS model "
                 "forecasts, not IMD cyclone warnings: for an active cyclone, IMD's bulletins are the official word. "
                 "Active and recent storm tracks are from IBTrACS."),
    }


BUILDERS = {"cyclones": _outlook_cyclones, "heatwaves": _outlook_heatwaves,
            "low_oxygen": _outlook_low_oxygen, "algal_blooms": _outlook_algal_blooms}


def get_outlook(hazard: str, *, day: date | None = None) -> dict:
    if hazard not in BUILDERS:
        raise UnknownHazardError(hazard)
    day = day or issue_day()
    key = (hazard, day.isoformat())
    if key in _outlooks:
        return _outlooks[key]
    with _lock(f"outlook:{hazard}:{day}"):
        if key in _outlooks:
            return _outlooks[key]
        try:
            result = BUILDERS[hazard](day)
        except (HazardError, heatwave_store.HeatwaveUnavailableError) as exc:
            raise HazardError(str(exc)) from exc
        except incois.ForecastUnavailableError as exc:
            raise HazardError(f"INCOIS forecast unavailable: {exc}") from exc
        result.update({"issued": day.isoformat(), "leads_h": LEADS_H})
        _outlooks[key] = result
        return result


# --- oil spill drift --------------------------------------------------------------------

def _wind_box(lat: float, lon: float, now: datetime, half_deg: float = 4.0):
    """INCOIS forecast 10 m wind (m/s) in a box around the release, 3-hourly for 72 h."""
    run = incois._run("wave", now.date())
    x, y = np.asarray(run["x"]), np.asarray(run["y"])
    i0, i1 = int(np.abs(x - (lon - half_deg)).argmin()), int(np.abs(x - (lon + half_deg)).argmin())
    j0, j1 = int(np.abs(y - (lat - half_deg)).argmin()), int(np.abs(y - (lat + half_deg)).argmin())
    a, _ = incois._window(run["times"], now)
    b = min(len(run["times"]) - 1, a + 25)
    s = 2
    q = f"[{a}:{b}][{j0}:{s}:{j1}][{i0}:{s}:{i1}]"
    data = incois._ascii(run["path"], f"UWND{q},VWND{q}")
    nt, ny, nx = b - a + 1, len(range(j0, j1 + 1, s)), len(range(i0, i1 + 1, s))
    u = np.asarray(data["UWND"], dtype="float64").reshape(nt, ny, nx)
    v = np.asarray(data["VWND"], dtype="float64").reshape(nt, ny, nx)
    u, v = np.where(u > -900, u, 0.0), np.where(v > -900, v, 0.0)
    hours = np.array([(t - now).total_seconds() / 3600 for t in run["times"][a:b + 1]])
    return hours, y[j0:j1 + 1:s], x[i0:i1 + 1:s], u, v


def _sample(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, la: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Nearest-cell sample of a 2-D field at many points; NaN outside."""
    j = np.clip(np.round((la - lats[0]) / (lats[1] - lats[0])).astype(int), 0, lats.size - 1)
    i = np.clip(np.round((lo - lons[0]) / (lons[1] - lons[0])).astype(int), 0, lons.size - 1)
    out = field[j, i]
    outside = (la < min(lats[0], lats[-1])) | (la > max(lats[0], lats[-1])) | (lo < min(lons[0], lons[-1])) | (lo > max(lons[0], lons[-1]))
    return np.where(outside, np.nan, out)


def oil_drift(lat: float, lon: float, *, particles: int = 200, hours: int = 72, seed: int = 7,
              now: datetime | None = None) -> dict:
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    day = now.date()
    if not (DOMAIN[1] <= lat <= DOMAIN[3] and DOMAIN[0] <= lon <= DOMAIN[2]):
        raise HazardError("the release point is outside the model domain (4–24°N, 62–96°E)")
    with NETCDF_LOCK, xr.open_dataset(product_file("cur", day)) as ds:
        cu = ds["uo"].isel(depth=0).values.astype("float64")
        cv = ds["vo"].isel(depth=0).values.astype("float64")
        clats, clons = ds["latitude"].values, ds["longitude"].values
        ctimes = np.asarray(ds["time"].values, dtype="datetime64[s]")
    chours = np.array([(np.datetime64(now.replace(tzinfo=None), "s") - t) / np.timedelta64(1, "h") * -1 for t in ctimes], dtype="float64")
    if np.isnan(_sample(cu[0], clats, clons, np.array([lat]), np.array([lon])))[0]:
        raise HazardError("the release point is on land in the ocean model; pick a point at sea")
    whours, wlats, wlons, wu, wv = _wind_box(lat, lon, now)

    rng = np.random.default_rng(seed)
    la = lat + rng.normal(0, 0.005, particles)
    lo = lon + rng.normal(0, 0.005, particles)
    beached = np.zeros(particles, dtype=bool)
    snapshots = {0: (la.copy(), lo.copy(), beached.copy())}
    dt = 3600.0
    sigma = math.sqrt(2 * DIFFUSIVITY_M2_S * dt)                       # metres per step
    for h in range(hours):
        ci = int(np.clip(np.searchsorted(chours, h, side="right") - 1, 0, len(chours) - 1))
        wi = int(np.clip(np.searchsorted(whours, h, side="right") - 1, 0, len(whours) - 1))
        u = _sample(cu[ci], clats, clons, la, lo) + WINDAGE * np.nan_to_num(_sample(wu[wi], wlats, wlons, la, lo))
        v = _sample(cv[ci], clats, clons, la, lo) + WINDAGE * np.nan_to_num(_sample(wv[wi], wlats, wlons, la, lo))
        move = ~beached & np.isfinite(u) & np.isfinite(v)
        dx = (np.nan_to_num(u) * dt + rng.normal(0, sigma, particles)) * move
        dy = (np.nan_to_num(v) * dt + rng.normal(0, sigma, particles)) * move
        new_la = la + dy / 111320.0
        new_lo = lo + dx / (111320.0 * np.cos(np.radians(la)))
        # A particle that would step onto land (no current there) is beached where it is.
        land = np.isnan(_sample(cu[ci], clats, clons, new_la, new_lo)) & move
        beached |= land
        la = np.where(move & ~land, new_la, la)
        lo = np.where(move & ~land, new_lo, lo)
        if (h + 1) % 6 == 0:
            snapshots[h + 1] = (la.copy(), lo.copy(), beached.copy())
    out = []
    for h in sorted(snapshots):
        a, b, beach = snapshots[h]
        cla, clo = float(np.mean(a)), float(np.mean(b))
        spread = float(np.sqrt(np.mean(((a - cla) * 111.32) ** 2 + ((b - clo) * 111.32 * math.cos(math.radians(cla))) ** 2)))
        dist = math.hypot((cla - lat) * 111.32, (clo - lon) * 111.32 * math.cos(math.radians(lat)))
        out.append({"hour": h, "time": (now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:00Z"),
                    "lat": [round(float(v), 4) for v in a], "lon": [round(float(v), 4) for v in b],
                    "centre": [round(cla, 4), round(clo, 4)], "spread_km": round(spread, 1),
                    "distance_km": round(dist, 1), "beached_fraction": round(float(beach.mean()), 3)})
    # A close-up window around the whole drift, 1.7:1 like the map, with the
    # model's land at full resolution for drawing it.
    all_la = np.concatenate([np.asarray(x["lat"]) for x in out])
    all_lo = np.concatenate([np.asarray(x["lon"]) for x in out])
    c_la, c_lo = (all_la.min() + all_la.max()) / 2, (all_lo.min() + all_lo.max()) / 2
    half_lat = max(0.5, (all_la.max() - all_la.min()) / 2 + 0.3, (all_lo.max() - all_lo.min()) / 2 / 1.7 + 0.3)
    view = {"latMin": c_la - half_lat, "latMax": c_la + half_lat, "lonMin": c_lo - half_lat * 1.7, "lonMax": c_lo + half_lat * 1.7}
    jj = np.flatnonzero((clats >= view["latMin"] - 0.1) & (clats <= view["latMax"] + 0.1))
    ii = np.flatnonzero((clons >= view["lonMin"] - 0.1) & (clons <= view["lonMax"] + 0.1))
    land = np.isnan(cu[0][np.ix_(jj, ii)]).astype("float64")
    return {
        "view": {k: round(float(v), 4) for k, v in view.items()},
        "land": {"grid": _grid(clats[jj], clons[ii]), "values": _flat(land, 0)},
        "release": {"lat": lat, "lon": lon, "time": now.strftime("%Y-%m-%dT%H:%M:00Z")},
        "particles": particles, "hours": hours, "snapshots": out,
        "method": {"currents": PRODUCTS["cur"][0] + " (surface, 6-hourly)",
                   "wind": "INCOIS WAVEWATCH III 10 m wind (3-hourly)", "windage": WINDAGE,
                   "diffusivity_m2_s": DIFFUSIVITY_M2_S, "time_step_s": dt},
        "note": ("A hypothetical release, not a reported spill: no live oil-spill feed exists for these seas. Particles move with "
                 "the forecast surface current plus 3% of the forecast wind, and spread by a random walk. Evaporation, "
                 "emulsification and clean-up are not modelled: this shows where oil would go, not how much would remain. "
                 "A particle that reaches the model's coastline stays there (beached)."),
    }
