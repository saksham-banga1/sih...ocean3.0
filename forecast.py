"""
72-hour ocean-state forecasts at seven INCOIS wave-rider buoy sites, from
INCOIS's own operational Ocean State Forecast (OSF).

INCOIS publishes the OSF model output behind its forecast maps on its THREDDS
server (https://incois.gov.in/thredds), openly readable over OPeNDAP:

  waves + wind  osf/ww3/rsmc_combined_ww3_YYYYMMDD.nc
                WAVEWATCH III: HS (significant wave height, m) and UWND/VWND,
                the 10 m wind that drives the wave model (m/s). 0.1 deg grid,
                3-hourly, seven days from 00 UTC the day after the run date.
  SST           osf/winds/SST_NIO_YYYYMMDD.nc
                SST (deg C) on a 1/12 deg grid, 3-hourly at hh:30 UTC.

These are the files INCOIS's own OSF page draws its maps from. A new run is
published daily; the newest of each kind is found by asking for today's date
and stepping back a day at a time. Only a few kilobytes are read per site: the
ASCII OPeNDAP response for one grid cell's time series.

Nothing is interpolated in time. Waves and wind come at 00, 03, ... UTC and SST
at 01:30, 04:30, ... UTC, each on its native steps. The SST run is published a
day behind the wave run, so its series can stop short of the 72 h horizon; the
response carries its own times, so where it ends is plain.

Sites. The seven WAMAN wave-rider buoys reporting to INCOIS's Ocean Observation
Network in the week of 20-27 September 2026 with the largest quality-controlled
records over 2009-2023 (Ocean Dynamics 76:16, 2026, Table 4). Positions are the
buoys' own, as INCOIS's network map reported them that week. Each series comes
from the nearest model cell with sea in it, and the response reports that cell
and its distance from the buoy.

These are INCOIS model forecasts, not the buoys' measurements, and not INCOIS's
official bulletins or warnings -- those are issued by INCOIS itself.

Caching. A site's forecast is kept for REFRESH_AFTER_SECONDS. If INCOIS cannot
be reached, the last good copy is served for up to STALE_LIMIT_SECONDS and
marked stale; past that, or with no copy at all, the request fails with an
explanation rather than showing made-up numbers.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

import cftime

logger = logging.getLogger(__name__)

THREDDS = "https://incois.gov.in/thredds/dodsC"
WAVE_PATH = "osf/ww3/rsmc_combined_ww3_{day}.nc"
SST_PATH = "osf/winds/SST_NIO_{day}.nc"
LOOKBACK_DAYS = 5          # how far back to look for the newest run
SEARCH_RADIUS_CELLS = 3    # how far from the buoy to look for a sea cell

HORIZON_HOURS = 72
REFRESH_AFTER_SECONDS = 3600
STALE_LIMIT_SECONDS = 12 * 3600
TIMEOUT_SECONDS = 30
MS_TO_KNOTS = 3600 / 1852

# The seven sites, in chip order: west coast north to south, then the east
# coast south to north, then the Andamans. "buoy" is INCOIS's own name for it.
STATIONS: dict[str, dict] = {
    "versova":       {"name": "Mumbai",        "buoy": "Versova",     "lat": 19.1353, "lon": 72.7418},
    "ratnagiri":     {"name": "Ratnagiri",     "buoy": "Ratnagiri",   "lat": 16.9760, "lon": 73.2550},
    "tuticorin":     {"name": "Tuticorin",     "buoy": "Tuticorin",   "lat": 8.8785,  "lon": 78.2929},
    "pondicherry":   {"name": "Puducherry",    "buoy": "Pondicherry", "lat": 11.9182, "lon": 79.8620},
    "visakhapatnam": {"name": "Visakhapatnam", "buoy": "Vizag",       "lat": 17.6314, "lon": 83.2522},
    "gopalpur":      {"name": "Gopalpur",      "buoy": "Gopalpur",    "lat": 19.2538, "lon": 84.9446},
    "port_blair":    {"name": "Port Blair",    "buoy": "PortBlair",   "lat": 11.6556, "lon": 92.7662},
}

SITES_NOTE = (
    "The seven INCOIS WAMAN wave-rider buoys reporting in the week of 20–27 September 2026 "
    "with the largest quality-controlled records (2009–2023). Positions as INCOIS's "
    "Ocean Observation Network reported them that week."
)

SOURCES = {
    "wave_height": {
        "model": "INCOIS WAVEWATCH III (Ocean State Forecast)", "variable": "HS", "units": "m",
        "description": "Significant wave height.",
    },
    "wind_speed": {
        "model": "INCOIS WAVEWATCH III wind forcing (Ocean State Forecast)", "variable": "UWND, VWND",
        "units": "kn", "description": "10 m wind speed, from the wind that drives the wave model.",
    },
    "sst": {
        "model": "INCOIS Ocean State Forecast SST", "variable": "SST", "units": "°C",
        "description": "Sea-surface temperature.",
    },
}

NOTE = (
    "INCOIS operational model forecasts, read from INCOIS's THREDDS server. They are not "
    "the buoys' measurements, and not INCOIS's official bulletins or warnings. Values are "
    "on the models' native 3-hourly steps; nothing is interpolated."
)


class ForecastUnavailableError(RuntimeError):
    """INCOIS could not be reached, or served nothing usable, and there is no recent copy."""


class UnknownStationError(KeyError):
    pass


# --- OPeNDAP, ASCII flavour ---------------------------------------------------

def _fetch_text(url: str) -> str:
    """GET a URL as text. Kept separate so the tests can replace it."""
    request = urllib.request.Request(url, headers={"User-Agent": "OCEANAO/0.1 (SIH ocean platform)"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", "replace")


def _ascii(path: str, query: str) -> dict[str, list[float]]:
    """Ask for `query` from `path` as OPeNDAP ASCII and parse it into flat lists.

    The response names each array ("HS.HS[25][1][1]", "HS.TIME[25]") and then
    lists its values, rows prefixed by their index ("[0][0], 1.02"). Keys here
    are the last part of each name, so a Grid's map vectors come back as TIME etc.
    """
    encoded = query.replace("[", "%5B").replace("]", "%5D")
    text = _fetch_text(f"{THREDDS}/{path}.ascii?{encoded}")
    if "-" * 45 not in text:
        raise ForecastUnavailableError(f"INCOIS THREDDS did not return data for {path}")
    body = text.split("-" * 45, 1)[1]
    arrays: dict[str, list[float]] = {}
    current = None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        header = re.match(r"^([A-Za-z0-9_.]+)\[", line)
        if header:
            current = header.group(1).split(".")[-1]
            arrays[current] = []
            continue
        if current is None:
            continue
        values = line.split(",")[1:] if line.startswith("[") else line.split(",")
        arrays[current].extend(float(v) for v in values if v.strip())
    return arrays


def _exists(path: str) -> bool:
    try:
        _fetch_text(f"{THREDDS}/{path}.dds")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404):
            return False
        raise


def _newest(template: str, today: date) -> tuple[str, date]:
    for back in range(LOOKBACK_DAYS + 1):
        day = today - timedelta(days=back)
        path = template.format(day=day.strftime("%Y%m%d"))
        if _exists(path):
            return path, day
    name = template.split("/")[-1].format(day="*")
    raise ForecastUnavailableError(f"INCOIS has published no {name} in the last {LOOKBACK_DAYS} days")


def _units(path: str, variable: str) -> str:
    das = _fetch_text(f"{THREDDS}/{path}.das")
    block = re.search(rf"\b{variable}\s*\{{(.*?)\}}", das, re.S)
    found = block and re.search(r'units\s+"([^"]+)"', block.group(1))
    if not found:
        raise ForecastUnavailableError(f"INCOIS {path} gives no units for {variable}")
    return found.group(1)


def _decode_times(values: list[float], units: str) -> list[datetime]:
    # CF "standard" calendar: mixed Julian/Gregorian. The wave file counts
    # "hours since 0001-01-01", and read as proleptic Gregorian it lands two
    # days late.
    dates = cftime.num2date(values, units, calendar="standard")
    return [datetime(d.year, d.month, d.day, d.hour, d.minute, tzinfo=timezone.utc) for d in dates]


# --- per-run state: file, axes, times, and each site's sea cell -----------------

KINDS = {
    "wave": {"template": WAVE_PATH, "x": "IOXAXIS", "y": "IOYAXIS", "t": "TIME",
             "probe": "HS[0][{y0}:{y1}][{x0}:{x1}]", "probe_var": "HS"},
    "sst": {"template": SST_PATH, "x": "LON", "y": "LAT", "t": "TAXIS",
            "probe": "SST[0][0][{y0}:{y1}][{x0}:{x1}]", "probe_var": "SST"},
}

_runs: dict[str, dict] = {}
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _is_value(v: float | None) -> bool:
    # INCOIS marks no-data as -1e34 (waves, wind) and -999.9 (SST).
    return v is not None and math.isfinite(v) and v > -900


def _run(kind: str, today: date) -> dict:
    """The newest run of one kind, with its axes and decoded times."""
    with _lock:
        run = _runs.get(kind)
    if run and run["checked"] == today and time.monotonic() - run["checked_at"] < REFRESH_AFTER_SECONDS:
        return run
    spec = KINDS[kind]
    path, day = _newest(spec["template"], today)
    if run and run["path"] == path:
        run = {**run, "checked": today, "checked_at": time.monotonic()}
    else:
        axes = _ascii(path, f"{spec['x']},{spec['y']},{spec['t']}")
        run = {
            "path": path, "day": day, "checked": today, "checked_at": time.monotonic(),
            "x": axes[spec["x"]], "y": axes[spec["y"]],
            "times": _decode_times(axes[spec["t"]], _units(path, spec["t"])),
            "cells": {},
        }
    with _lock:
        _runs[kind] = run
    return run


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _nearest_index(axis: list[float], value: float) -> int:
    return min(range(len(axis)), key=lambda i: abs(axis[i] - value))


def _sea_cell(kind: str, run: dict, key: str, site: dict | None = None) -> dict:
    """The model cell nearest the buoy (or any given site) that has sea in it. The
    buoys sit a few km offshore, so the cell containing one can be land in the model."""
    if key in run["cells"]:
        return run["cells"][key]
    spec, site = KINDS[kind], site or STATIONS[key]
    i, j = _nearest_index(run["x"], site["lon"]), _nearest_index(run["y"], site["lat"])
    r = SEARCH_RADIUS_CELLS
    x0, x1 = max(0, i - r), min(len(run["x"]) - 1, i + r)
    y0, y1 = max(0, j - r), min(len(run["y"]) - 1, j + r)
    values = _ascii(run["path"], spec["probe"].format(x0=x0, x1=x1, y0=y0, y1=y1))[spec["probe_var"]]
    width = x1 - x0 + 1
    best = None
    for n, v in enumerate(values):
        if not _is_value(v):
            continue
        yy, xx = y0 + n // width, x0 + n % width
        d = _km(site["lat"], site["lon"], run["y"][yy], run["x"][xx])
        if best is None or d < best["distance_km"]:
            best = {"i": xx, "j": yy, "lat": round(run["y"][yy], 4), "lon": round(run["x"][xx], 4),
                    "distance_km": round(d, 1)}
    if best is None:
        raise ForecastUnavailableError(
            f"the INCOIS {kind} model has no sea cell within {r} cells of the {site.get('buoy', site.get('name', key))}")
    run["cells"][key] = best
    return best


def _window(times: list[datetime], now: datetime) -> tuple[int, int]:
    """Indices from the last step at or before now to the last within now + 72 h."""
    end = now + timedelta(hours=HORIZON_HOURS)
    first = max((k for k, t in enumerate(times) if t <= now), default=0)
    last = max((k for k, t in enumerate(times) if t <= end), default=-1)
    return first, last


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:00Z")


def _clean(v: float, digits: int) -> float | None:
    return round(v, digits) if _is_value(v) else None


def _fetch(key: str, now: datetime) -> dict:
    site = STATIONS[key]
    today = now.date()

    wave_run = _run("wave", today)
    cell = _sea_cell("wave", wave_run, key)
    a, b = _window(wave_run["times"], now)
    if b < a:
        raise ForecastUnavailableError(f"INCOIS wave run {wave_run['path']} has no steps in the next {HORIZON_HOURS} h")
    at = f"[{a}:{b}][{cell['j']}][{cell['i']}]"
    wave = _ascii(wave_run["path"], f"HS{at},UWND{at},VWND{at}")

    heights = [_clean(v, 2) for v in wave["HS"]]
    if not any(h is not None for h in heights):
        raise ForecastUnavailableError(f"the INCOIS wave model has no wave height at the {site['buoy']} buoy")
    speeds, directions = [], []
    for u, v in zip(wave["UWND"], wave["VWND"]):
        if _is_value(u) and _is_value(v):
            speeds.append(round(math.hypot(u, v) * MS_TO_KNOTS, 1))
            # The direction the wind blows FROM, clockwise from north.
            directions.append(round(math.degrees(math.atan2(-u, -v)) % 360))
        else:
            speeds.append(None)
            directions.append(None)

    sst_run = _run("sst", today)
    s_cell = _sea_cell("sst", sst_run, key)
    sa, sb = _window(sst_run["times"], now)
    if sb >= sa:
        sst = _ascii(sst_run["path"], f"SST[{sa}:{sb}][0][{s_cell['j']}][{s_cell['i']}]")["SST"]
        sst_times = [_iso(t) for t in sst_run["times"][sa:sb + 1]]
        sst_values = [_clean(v, 2) for v in sst]
    else:
        sst_times, sst_values = [], []

    return {
        "station": key,
        "name": site["name"],
        "buoy": {"name": site["buoy"], "lat": site["lat"], "lon": site["lon"]},
        "cells": {
            "wave": {k: cell[k] for k in ("lat", "lon", "distance_km")},
            "sst": {k: s_cell[k] for k in ("lat", "lon", "distance_km")},
        },
        "wave": {
            "times": [_iso(t) for t in wave_run["times"][a:b + 1]],
            "wave_height": heights,
            "wind_speed": speeds,
            "wind_direction": directions,
        },
        "sst": {"times": sst_times, "values": sst_values},
        "runs": {
            "wave": {"file": wave_run["path"], "run_date": wave_run["day"].isoformat()},
            "sst": {"file": sst_run["path"], "run_date": sst_run["day"].isoformat()},
        },
        "horizon_end": _iso(now + timedelta(hours=HORIZON_HOURS)),
        "sources": SOURCES,
        "sites_note": SITES_NOTE,
        "note": NOTE,
        "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stale": False,
    }


def get_forecast(key: str, *, now: datetime | None = None) -> dict:
    """The next 72 h at one INCOIS wave-rider buoy site, from INCOIS's OSF."""
    if key not in STATIONS:
        raise UnknownStationError(key)
    clock = time.monotonic()
    with _lock:
        cached = _cache.get(key)
    if cached and clock - cached[0] < REFRESH_AFTER_SECONDS:
        return cached[1]
    try:
        payload = _fetch(key, now or datetime.now(timezone.utc))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, ForecastUnavailableError) as exc:
        if isinstance(exc, ForecastUnavailableError):
            reason = str(exc)
        elif isinstance(exc, (ValueError, KeyError)):
            reason = f"INCOIS THREDDS returned something unexpected ({exc!r})"
        else:
            reason = f"INCOIS THREDDS is not reachable ({exc})"
        if cached and clock - cached[0] < STALE_LIMIT_SECONDS:
            logger.warning("forecast %s: %s; serving the copy from %s", key, reason, cached[1]["fetched_at"])
            return {**cached[1], "stale": True}
        raise ForecastUnavailableError(reason) from exc
    with _lock:
        _cache[key] = (clock, payload)
    return payload


_now_cache: dict[str, tuple[float, dict]] = {}


def conditions_now(key: str, lat: float, lon: float, *, now: datetime | None = None) -> dict:
    """INCOIS OSF sea-surface temperature and wave height for the current step at
    any point in the domain, from the same runs the station forecasts read."""
    clock = time.monotonic()
    cached = _now_cache.get(key)
    if cached and clock - cached[0] < REFRESH_AFTER_SECONDS:
        return cached[1]
    now = now or datetime.now(timezone.utc)
    site = {"name": key, "lat": lat, "lon": lon}
    wave_run = _run("wave", now.date())
    cell = _sea_cell("wave", wave_run, f"point:{key}", site)
    a, _ = _window(wave_run["times"], now)
    hs = _ascii(wave_run["path"], f"HS[{a}][{cell['j']}][{cell['i']}]")["HS"][0]
    sst_run = _run("sst", now.date())
    s_cell = _sea_cell("sst", sst_run, f"point:{key}", site)
    sa, _ = _window(sst_run["times"], now)
    sst = _ascii(sst_run["path"], f"SST[{sa}][0][{s_cell['j']}][{s_cell['i']}]")["SST"][0]
    result = {
        "wave_height_m": _clean(hs, 2), "wave_time": _iso(wave_run["times"][a]),
        "sst_c": _clean(sst, 2), "sst_time": _iso(sst_run["times"][sa]),
        "wave_cell_km": cell["distance_km"], "sst_cell_km": s_cell["distance_km"],
        "source": "INCOIS Ocean State Forecast (WAVEWATCH III and OSF SST), nearest sea cell",
        "runs": {"wave": wave_run["day"].isoformat(), "sst": sst_run["day"].isoformat()},
    }
    _now_cache[key] = (clock, result)
    return result


def list_stations() -> list[dict]:
    return [{"station": key, **value} for key, value in STATIONS.items()]


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        _runs.clear()
        _now_cache.clear()
