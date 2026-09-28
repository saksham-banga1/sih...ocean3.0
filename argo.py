"""
Fetch real Argo profiling float observations via argopy.

Replaces the hardcoded `argoFloats` array in main.js (ARGO-2902143,
ARGO-2903381) with live data from the Argo GDAC, over the same basin boxes the
Copernicus model downloads use (app/services/regions.py).

Two things worth knowing about the underlying data:

* Argo reports **pressure in decibars**, not depth in metres. `_pressure_to_depth`
  converts with the UNESCO / Fofonoff & Millard (1983) formula, which needs
  latitude because gravity varies with it. Treating dbar as metres directly is
  off by roughly 1-2%.
* Every measurement carries a QC flag. Only flags 1 (good) and 2 (probably good)
  are kept; anything else is dropped rather than shown to the user.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Imported at module load, not lazily inside the fetch functions. FastAPI runs
# sync endpoints in a threadpool, so two concurrent first-time requests would
# both trigger argopy's import and deadlock on Python's module lock
# (_DeadlockError on argopy.xarray), silently failing the sensor list.
from argopy import DataFetcher

from app.services.regions import PRESET_BBOXES, get_bbox  # noqa: F401 - re-exported

# QC flags considered trustworthy for display (Argo reference table 2).
GOOD_QC_FLAGS = (1, 2)

# How far back to look when deciding a float is "active".
DEFAULT_ACTIVE_DAYS = 60

# Argo floats profile to ~2000 m, matching the model download depth range.
DEFAULT_MAX_PRESSURE = 2000.0

# How far back a drift trajectory reaches. Deliberately much longer than
# DEFAULT_ACTIVE_DAYS, and deliberately a separate number: 60 days decides
# whether a float is still reporting, which has to stay a recent judgement,
# while the drift path is history and only gets more useful with depth.
# Widening this is affordable because fetch_float_tracks reads the profile INDEX
# -- date, latitude, longitude, cycle -- and transfers no measurements at all.
#
# Measured against the erddap source over the full_domain box:
#     60 d -> 13.8 s,   497 rows, 109 floats, median  5 cycles/float
#    180 d -> 21.3 s,  1577 rows, 113 floats, median 16 cycles/float
#    365 d -> 39.1 s,  3119 rows, 124 floats, median 33 cycles/float
#    730 d -> ErddapServerError after 15 s; the server refuses the span
# 365 is therefore the practical ceiling, and at ~39 s it is far too slow to sit
# in front of the sensor list -- hence its own endpoint, fetched after the
# markers are already on the globe.
DEFAULT_TRACK_DAYS = 365

# Stated on every trajectory response. The API is a public surface: a consumer
# that never sees the UI still has to be told that the geometry between two
# surfacings is drawn, not observed.
TRACK_POSITION_NOTE = (
    "Positions are recorded surfacings, typically ~10 days apart, oldest first. "
    "The float drifts at depth between them, so any line joining two positions is "
    "display geometry, not a measured path. The earliest position is the earliest "
    "within this box and window, not a deployment site. last_cycle may exceed the "
    "float's current cycle elsewhere in this API, because the GDAC publishes a "
    "profile's position before its measurements."
)

# Ceiling on the positions in one drift track. At the usual ~10 day cycle a year
# is ~36 fixes, so this bites only for unusually fast cyclers; it exists to bound
# the payload, not to trim the history.
MAX_TRACK_POINTS = 120

# Network fetches take seconds, so hold results briefly. Argo data updates on the
# order of days, so a few minutes of staleness costs nothing.
CACHE_TTL_SECONDS = 600

_cache: dict[Any, tuple[float, Any]] = {}
_cache_lock = threading.Lock()

# --- persistent fallback ----------------------------------------------------
# The in-memory TTL above makes repeat requests fast. This disk layer is a
# different job: it keeps the last SUCCESSFUL REAL response so the app still
# serves genuine Argo data when the GDAC is unreachable. It never holds
# anything synthetic -- only payloads that came back from a live fetch.
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DISK_CACHE_DIR = BACKEND_DIR.parent / "data" / "observations" / "argo_cache"

# Keep the directory bounded; entries are small JSON documents.
DISK_CACHE_MAX_FILES = 200


def _disk_cache_path(key: Any) -> Path:
    """Stable filename for a cache key, readable enough to inspect by hand."""
    text = "|".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
    digest = hashlib.sha1(text.encode()).hexdigest()[:16]
    label = re.sub(r"[^a-zA-Z0-9]+", "-", text)[:60].strip("-").lower()
    return DISK_CACHE_DIR / f"{label}-{digest}.json"


def _write_disk_cache(key: Any, value: Any) -> None:
    """Persist a live result. Atomic: a crash mid-write cannot corrupt the file.

    The payload is written to a temporary file in the same directory and then
    os.replace()d onto the target, which is atomic on POSIX. Readers therefore
    see either the old complete file or the new complete file, never a partial.
    """
    path = _disk_cache_path(key)
    try:
        DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": str(key),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": value,
        }
        # Same directory so os.replace stays on one filesystem.
        handle = tempfile.NamedTemporaryFile(
            "w", dir=DISK_CACHE_DIR, prefix=".tmp-", suffix=".json", delete=False
        )
        try:
            with handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())  # durable before the rename
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        _prune_disk_cache()
    except Exception as exc:  # noqa: BLE001 - caching must never break a good fetch
        print(f"argo disk cache: could not persist {key}: {exc}")


def _read_disk_cache(key: Any) -> tuple[Any, str] | None:
    """Last successful live payload for this key, as (data, fetched_at)."""
    path = _disk_cache_path(key)
    if not path.is_file():
        return None
    try:
        with path.open() as handle:
            payload = json.load(handle)
        return payload["data"], payload["fetched_at"]
    except Exception as exc:  # noqa: BLE001 - a damaged entry must not be fatal
        print(f"argo disk cache: ignoring unreadable {path.name}: {exc}")
        return None


def _prune_disk_cache() -> None:
    """Drop the oldest entries once the directory grows past its cap."""
    try:
        files = sorted(
            DISK_CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for stale in files[DISK_CACHE_MAX_FILES:]:
            stale.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def disk_cache_status() -> dict:
    """What the persistent fallback currently holds."""
    try:
        files = list(DISK_CACHE_DIR.glob("*.json"))
    except Exception:  # noqa: BLE001
        files = []
    newest = max((f.stat().st_mtime for f in files), default=None)
    return {
        "directory": str(DISK_CACHE_DIR),
        "entries": len(files),
        "last_synced_at": (
            datetime.fromtimestamp(newest, timezone.utc).isoformat(timespec="seconds")
            if newest
            else None
        ),
    }


class FloatNotFoundError(LookupError):
    """No Argo float matched the requested id, or it has no usable profile."""


class ArgoUnavailableError(RuntimeError):
    """The Argo GDAC could not be reached. Upstream problem, not a bad request."""


def _cached(key: Any, produce) -> tuple[Any, dict]:
    """TTL memo in front, persistent real-data fallback behind.

    Returns (value, provenance) where provenance carries source and
    last_synced_at. Order of preference:
      1. in-memory entry inside its TTL      -> source "live"
      2. a fresh live fetch                  -> source "live", persisted to disk
      3. the last successful live payload    -> source "cached"
    Synthetic data is never produced here; if there is no cache either, the
    upstream error propagates so the caller can report it honestly.
    """
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1], {
                "source": "live",
                "last_synced_at": datetime.fromtimestamp(
                    hit[0], timezone.utc
                ).isoformat(timespec="seconds"),
            }

    try:
        value = produce()  # outside the lock: this is a network round-trip
    except ArgoUnavailableError:
        # GDAC unreachable. Serve the last REAL response we stored, if any.
        fallback = _read_disk_cache(key)
        if fallback is None:
            raise
        data, fetched_at = fallback
        return data, {"source": "cached", "last_synced_at": fetched_at}

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_disk_cache(key, value)

    with _cache_lock:
        _cache[key] = (time.time(), value)
    return value, {"source": "live", "last_synced_at": fetched_at}


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _pressure_to_depth(pressure_dbar, latitude: float):
    """Convert Argo pressure (dbar) to depth (m), UNESCO / Fofonoff & Millard 1983.

    Gravity varies with latitude, so the same pressure sits at a slightly
    different depth at the equator than near the poles.
    """
    p = np.asarray(pressure_dbar, dtype=float)
    x = math.sin(math.radians(abs(latitude))) ** 2
    gravity = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * p
    numerator = (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p
    return numerator / gravity


def normalise_float_id(float_id: str | int) -> int:
    """Accept 'ARGO-2902143', '2902143' or 2902143 and return the WMO int.

    Raises ValueError for a malformed id -- that is bad input from the caller,
    distinct from FloatNotFoundError, which means a well-formed id that no float
    matches.
    """
    if isinstance(float_id, (int, np.integer)):
        return int(float_id)

    text = str(float_id).strip().upper()
    if text.startswith("ARGO-"):
        text = text[len("ARGO-"):]
    if not text.isdigit():
        raise ValueError(
            f"invalid float id {float_id!r} -- expected a WMO number like 2902143 "
            f"or 'ARGO-2902143'"
        )
    return int(text)


def format_float_id(wmo: int) -> str:
    """Render a WMO number the way main.js already labels floats."""
    return f"ARGO-{int(wmo)}"


def fetch_floats_in_bbox(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    days: int = DEFAULT_ACTIVE_DAYS,
    max_pressure: float = DEFAULT_MAX_PRESSURE,
    with_provenance: bool = False,
):
    """List active Argo floats in a box, one entry each at its most recent cycle.

    Arguments are in (min_lon, min_lat, max_lon, max_lat) order to match
    app/services/regions.py and Cesium's Rectangle.fromDegrees. argopy wants
    lon pair then lat pair, so they are reordered internally.
    """
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")

    end = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    key = ("bbox", lon_min, lat_min, lon_max, lat_max, days, max_pressure, str(start.date()))

    def produce() -> list[dict]:
        # argopy box order: [lon_min, lon_max, lat_min, lat_max, p_min, p_max, t0, t1]
        box = [
            lon_min, lon_max,
            lat_min, lat_max,
            0.0, max_pressure,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        ]
        # to_index() returns just the profile metadata (date/lat/lon/wmo/cyc),
        # which is far cheaper than pulling every measurement.
        try:
            index = DataFetcher(src="erddap").region(box).to_index()
        except FileNotFoundError:
            # No profiles in this box/window -- an empty region, not an error.
            return []
        except Exception as exc:
            raise ArgoUnavailableError(
                f"could not reach the Argo GDAC: {type(exc).__name__}: {exc}"
            ) from exc

        if index is None or len(index) == 0:
            return []

        # Most recent cycle per float.
        latest = index.sort_values("date").groupby("wmo", as_index=False).last()

        floats = [
            {
                "id": format_float_id(row.wmo),
                "wmo": int(row.wmo),
                "lat": round(float(row.latitude), 4),
                "lon": round(float(row.longitude), 4),
                "cycle": int(row.cyc),
                "last_seen": pd.Timestamp(row.date).isoformat(),
            }
            for row in latest.itertuples(index=False)
        ]
        floats.sort(key=lambda f: f["wmo"])
        return floats

    records, provenance = _cached(key, produce)
    return (records, provenance) if with_provenance else records


def fetch_surface_observations(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    days: int = DEFAULT_ACTIVE_DAYS,
    surface_pressure: float = 10.0,
    with_provenance: bool = False,
):
    """Surface temp/salinity for every float in a box, in ONE network call.

    Calling get_float_profile() per float costs a round-trip each -- ~5s x N,
    which is minutes for a basin. This pulls only the top `surface_pressure`
    decibars for the whole region in a single request and reduces per float,
    which is all the sensor list needs.
    """
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")

    end = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    key = ("surface", lon_min, lat_min, lon_max, lat_max, days, surface_pressure, str(start.date()))

    def produce() -> list[dict]:
        box = [
            lon_min, lon_max,
            lat_min, lat_max,
            0.0, surface_pressure,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        ]
        try:
            ds = DataFetcher(src="erddap").region(box).to_xarray()
        except FileNotFoundError:
            return []
        except Exception as exc:
            raise ArgoUnavailableError(
                f"could not reach the Argo GDAC: {type(exc).__name__}: {exc}"
            ) from exc

        if ds is None or ds.sizes.get("N_POINTS", 0) == 0:
            return []

        good = (
            np.isin(ds.TEMP_QC.values, GOOD_QC_FLAGS)
            & np.isin(ds.PSAL_QC.values, GOOD_QC_FLAGS)
            & np.isin(ds.PRES_QC.values, GOOD_QC_FLAGS)
            & np.isfinite(ds.TEMP.values)
            & np.isfinite(ds.PSAL.values)
            & np.isfinite(ds.PRES.values)
        )
        if not good.any():
            return []

        frame = pd.DataFrame(
            {
                "wmo": ds.PLATFORM_NUMBER.values[good].astype(int),
                "cycle": ds.CYCLE_NUMBER.values[good].astype(int),
                "lat": ds.LATITUDE.values[good].astype(float),
                "lon": ds.LONGITUDE.values[good].astype(float),
                "time": ds.TIME.values[good],
                "pres": ds.PRES.values[good].astype(float),
                "temp": ds.TEMP.values[good].astype(float),
                "psal": ds.PSAL.values[good].astype(float),
            }
        )

        frame = frame.sort_values(["wmo", "cycle", "pres"])

        # One surfacing position per cycle -- the float's real drift track.
        per_cycle = frame.groupby(["wmo", "cycle"], as_index=False).first()
        tracks: dict[int, list[list[float]]] = {}
        for wmo, group in per_cycle.sort_values("cycle").groupby("wmo"):
            tracks[int(wmo)] = [
                [round(float(r.lon), 4), round(float(r.lat), 4)]
                for r in group.itertuples(index=False)
            ][-MAX_TRACK_POINTS:]

        # Latest cycle per float, then its shallowest measurement.
        latest_cycle = frame.groupby("wmo")["cycle"].transform("max")
        frame = frame[frame["cycle"] == latest_cycle]
        shallowest = frame.groupby("wmo", as_index=False).first()

        return [
            {
                "id": format_float_id(row.wmo),
                "wmo": int(row.wmo),
                "lat": round(float(row.lat), 4),
                "lon": round(float(row.lon), 4),
                "cycle": int(row.cycle),
                "last_seen": pd.Timestamp(row.time).isoformat(),
                "depth": round(float(_pressure_to_depth(row.pres, float(row.lat))), 2),
                "temp": round(float(row.temp), 3),
                "salinity": round(float(row.psal), 3),
                # Real surfacing positions, oldest first. Replaces the fabricated
                # driftPath arrays that were hardcoded in main.js. Bounded by
                # THIS call's window, which is about recency rather than history,
                # so it is short -- enough to draw something immediately. For the
                # full trajectory see fetch_float_tracks.
                "track": tracks.get(int(row.wmo), []),
            }
            for row in shallowest.itertuples(index=False)
        ]

    records, provenance = _cached(key, produce)
    return (records, provenance) if with_provenance else records


def fetch_float_tracks(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    days: int = DEFAULT_TRACK_DAYS,
    with_provenance: bool = False,
):
    """Drift trajectories for every float in a box: one surfacing per cycle.

    Kept separate from fetch_surface_observations on purpose. That call answers
    "which floats are reporting now, and what did they last measure", so its
    window has to stay short or a long-silent float would be listed as current.
    This one answers "where has each float been", which only improves with more
    history. It reads to_index(), i.e. profile metadata, so a year of trajectory
    costs a fraction of what a year of measurements would.

    Three honest limits, all of which the UI states rather than hides:

    * The positions are surfacings, typically ~10 days apart. What the float did
      between two of them is not recorded anywhere, so the straight segment a
      map draws between them is display geometry, not a measured path.
    * The query is bounded by the box, so a float that drifted in from outside
      has its track clipped at the boundary. The earliest position in a track is
      the earliest IN THIS BOX AND WINDOW, never a deployment site.
    * The index publishes a profile's position before its measurements, so
      last_cycle here routinely runs ahead of the same float's cycle in
      fetch_surface_observations. That is why the cycle span is returned: a
      caller can then say the line ends past the marker on purpose.

    Returns, keyed by WMO number as a string because this value is persisted as
    JSON and JSON has no integer keys:

        {"1902670": {"positions": [[lon, lat], ...],   # oldest first
                     "first_cycle": 75,
                     "last_cycle": 109}}
    """
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")

    end = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    key = ("tracks", lon_min, lat_min, lon_max, lat_max, days, str(start.date()))

    def produce() -> dict[str, list[list[float]]]:
        box = [
            lon_min, lon_max,
            lat_min, lat_max,
            0.0, DEFAULT_MAX_PRESSURE,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        ]
        try:
            index = DataFetcher(src="erddap").region(box).to_index()
        except FileNotFoundError:
            # No profiles in this box/window -- an empty region, not an error.
            return {}
        except Exception as exc:
            raise ArgoUnavailableError(
                f"could not reach the Argo GDAC: {type(exc).__name__}: {exc}"
            ) from exc

        if index is None or len(index) == 0:
            return {}

        frame = index.dropna(subset=["wmo", "cyc", "latitude", "longitude", "date"])
        if frame.empty:
            return {}

        # One position per cycle, ordered BY CYCLE rather than by date. The
        # cycle number is the float's own count of surfacings, so it is what
        # defines the order it visited these places; timestamps occasionally
        # disagree with it (the index has floats whose cycle N is dated after
        # cycle N+1), and sorting on those would hand back a track that doubles
        # back on itself. Sorting on cycle also makes the returned list strictly
        # increasing, which is what lets a consumer read a jump as a real gap.
        # Within one cycle, the earliest row wins, so a re-upload does not move
        # the position.
        frame = frame.sort_values(["wmo", "cyc", "date"]).drop_duplicates(
            subset=["wmo", "cyc"], keep="first"
        )

        tracks: dict[str, dict] = {}
        for wmo, group in frame.groupby("wmo"):
            kept = group.tail(MAX_TRACK_POINTS)
            # A single fix draws no line; do not ship it as if it were a track.
            if len(kept) < 2:
                continue
            tracks[str(int(wmo))] = {
                "positions": [
                    [round(float(row.longitude), 4), round(float(row.latitude), 4)]
                    for row in kept.itertuples(index=False)
                ],
                # The cycle each position belongs to, same order. Consecutive
                # numbers mean consecutive surfacings; a jump means cycles are
                # missing, because the float was outside this box or those
                # profiles are not in the index. A caller must break its line
                # there -- the float did not travel in a straight line across a
                # gap it has no fixes for.
                "cycles": [int(c) for c in kept["cyc"].tolist()],
                # Cycle numbers of the kept span. They matter because the index
                # publishes a profile's POSITION before its MEASUREMENTS appear,
                # so a track routinely reaches cycles that the surface
                # observation feed has no data for yet. A consumer comparing
                # last_cycle against a float's current cycle can say so instead
                # of looking like it drew the line past the marker by mistake.
                "first_cycle": int(kept["cyc"].iloc[0]),
                "last_cycle": int(kept["cyc"].iloc[-1]),
            }
        return tracks

    records, provenance = _cached(key, produce)
    return (records, provenance) if with_provenance else records


def fetch_tracks_in_region(region: str, **kwargs):
    """fetch_float_tracks for a named preset ('bay_of_bengal', 'arabian_sea')."""
    return fetch_float_tracks(*get_bbox(region), **kwargs)


def fetch_floats_in_region(region: str, **kwargs) -> list[dict]:
    """fetch_floats_in_bbox for a named preset ('bay_of_bengal', 'arabian_sea')."""
    return fetch_floats_in_bbox(*get_bbox(region), **kwargs)


def get_float_profile(
    float_id: str | int, *, max_pressure: float = DEFAULT_MAX_PRESSURE
) -> dict:
    """Return the most recent temperature/salinity profile for one float.

    {"id", "lat", "lon", "cycle", "profile": [{"depth", "temp", "salinity"}, ...]}
    sorted shallowest first, QC-screened, with pressure converted to depth.
    """
    wmo = normalise_float_id(float_id)
    key = ("float", wmo, max_pressure)

    def produce() -> dict:
        try:
            ds = DataFetcher(src="erddap").float(wmo).to_xarray()
        except FileNotFoundError as exc:
            # argopy surfaces an unknown WMO as FileNotFoundError on the query URL.
            raise FloatNotFoundError(f"no Argo float with WMO {wmo}") from exc
        except Exception as exc:  # network, TLS, timeout -- upstream, not the caller
            raise ArgoUnavailableError(
                f"could not reach the Argo GDAC: {type(exc).__name__}: {exc}"
            ) from exc

        if ds is None or ds.sizes.get("N_POINTS", 0) == 0:
            raise FloatNotFoundError(f"Argo float {wmo} returned no measurements")

        # Keep only the latest cycle.
        cycles = ds.CYCLE_NUMBER.values
        latest_cycle = int(np.nanmax(cycles))
        ds = ds.isel(N_POINTS=np.flatnonzero(cycles == latest_cycle))

        pres = ds.PRES.values.astype(float)
        temp = ds.TEMP.values.astype(float)
        psal = ds.PSAL.values.astype(float)

        good = (
            np.isin(ds.PRES_QC.values, GOOD_QC_FLAGS)
            & np.isin(ds.TEMP_QC.values, GOOD_QC_FLAGS)
            & np.isin(ds.PSAL_QC.values, GOOD_QC_FLAGS)
            & np.isfinite(pres)
            & np.isfinite(temp)
            & np.isfinite(psal)
            & (pres <= max_pressure)
        )

        if not good.any():
            raise FloatNotFoundError(
                f"Argo float {wmo} cycle {latest_cycle} has no QC-passing measurements"
            )

        lat = float(np.nanmean(ds.LATITUDE.values[good]))
        lon = float(np.nanmean(ds.LONGITUDE.values[good]))
        depth = _pressure_to_depth(pres[good], lat)

        order = np.argsort(depth)  # shallowest first, as a profile reads
        profile = [
            {
                "depth": round(float(d), 2),
                "temp": round(float(t), 3),
                "salinity": round(float(s), 3),
            }
            for d, t, s in zip(depth[order], temp[good][order], psal[good][order])
        ]

        times = ds.TIME.values[good]
        return {
            "id": format_float_id(wmo),
            "wmo": wmo,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "cycle": latest_cycle,
            "time": str(np.max(times))[:19],
            "profile": profile,
        }

    profile, provenance = _cached(key, produce)
    return {**profile, **provenance}
