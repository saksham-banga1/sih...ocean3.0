"""
Real tropical cyclone tracks for the North Indian Ocean, from NOAA's IBTrACS.

IBTrACS (the International Best Track Archive for Climate Stewardship) merges
the official track data of every regional centre. For this basin that centre is
RSMC New Delhi -- the India Meteorological Department (IMD) -- so the winds,
pressures and storm grades served here are IMD's own, as IBTrACS publishes them.

What is served:

  active  every North Indian system IBTrACS lists as active, at any strength:
          a depression today can be a cyclone tomorrow.
  recent  past storms that reached cyclonic-storm strength (IMD grade CS or
          stronger), crossed the map area, and ended within the last N days.
          They are context, not current conditions, and carry their dates so
          they cannot be mistaken for either.

Nothing here is invented or forecast. IBTrACS adds storms with a delay, so a
storm missing from these tracks may simply not be in the archive yet.

Caching. The recent-storms file is ~10 MB and has taken over 30 s to download
on a slow link, so both files live on disk and are refreshed in the background
once they age past their refresh interval: a request is answered from disk at
once and never waits on a refresh. Only the very first request, before anything
is cached, waits for the download. Refreshes send If-Modified-Since, so an
unchanged file costs a few hundred bytes. If NOAA cannot be reached, the last
good copy keeps being served and is marked stale; if there has never been a
copy, the request fails with an explanation rather than showing a made-up track.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.services.regions import PRESET_BBOXES

logger = logging.getLogger(__name__)

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/"
)
SOURCE = "NOAA NCEI IBTrACS v04r01"

FILES = {
    "recent": "ibtracs.last3years.list.v04r01.csv",  # every storm of the last three years
    "active": "ibtracs.ACTIVE.list.v04r01.csv",      # storms NOAA currently lists as active
}
REFRESH_AFTER_SECONDS = {"recent": 12 * 3600, "active": 3600}
DOWNLOAD_TIMEOUT_SECONDS = 180

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BACKEND_DIR.parent / "data" / "observations" / "ibtracs_cache"

BASIN = "NI"
MAP_BBOX = PRESET_BBOXES["full_domain"]  # (min_lon, min_lat, max_lon, max_lat)
DEFAULT_WINDOW_DAYS = 365
MAX_WINDOW_DAYS = 1095  # the recent-storms file only reaches back three years

# IMD's intensity scale, weakest first.
IMD_GRADES = ("D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS")
IMD_GRADE_NAMES = {
    "D": "Depression",
    "DD": "Deep Depression",
    "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm",
    "VSCS": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm",
    "SuCS": "Super Cyclonic Storm",
}
_GRADE_BY_UPPER = {grade.upper(): grade for grade in IMD_GRADES}
CYCLONIC_STORM_KT = 34  # IMD's lower bound for a cyclonic storm

SUBBASIN_NAMES = {"AS": "Arabian Sea", "BB": "Bay of Bengal"}

WIND_NOTE = (
    "Winds are IMD (RSMC New Delhi) maximum sustained winds averaged over 3 minutes, "
    "in knots, as published in IBTrACS. A position without an IMD report has no wind "
    "value; other agencies' winds, which are averaged differently, are not mixed in."
)
SELECTION_NOTE = (
    "Active: every North Indian Ocean system IBTrACS lists as active, at any strength. "
    "Past: storms that reached cyclonic-storm strength or stronger (IMD grade CS+), "
    "crossed the map area ({min_lon:g}–{max_lon:g}°E, {min_lat:g}–{max_lat:g}°N) and "
    "ended within the last {days} days. Past storms are context, not current conditions."
)

_COLUMNS = ("SID", "BASIN", "SUBBASIN", "NAME", "ISO_TIME", "LAT", "LON",
            "WMO_WIND", "WMO_PRES", "TRACK_TYPE", "NEWDELHI_GRADE")

_download_locks = {kind: threading.Lock() for kind in FILES}
_refresh_guard = threading.Lock()
_refreshing: set[str] = set()
_parsed: dict[tuple, pd.DataFrame] = {}
_parsed_lock = threading.Lock()


class CycloneDataUnavailableError(RuntimeError):
    """No usable copy of the IBTrACS files, and NOAA could not be reached."""


# --------------------------------------------------------------------------
# Download and disk cache
# --------------------------------------------------------------------------


def _open_url(request: urllib.request.Request, timeout: float):
    """The one network call, kept separate so tests can cut the network."""
    return urllib.request.urlopen(request, timeout=timeout)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _paths(kind: str) -> tuple[Path, Path]:
    name = FILES[kind]
    return CACHE_DIR / name, CACHE_DIR / f"{name}.meta.json"


def _read_meta(meta_path: Path) -> dict:
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return {}


def _write_json_atomic(path: Path, payload: dict) -> None:
    handle = tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-", suffix=".json", delete=False)
    try:
        with handle:
            json.dump(payload, handle)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _download(kind: str) -> None:
    """Fetch one IBTrACS file, or confirm the cached copy is still current.

    Written to a temporary file and swapped in with os.replace, so a reader never
    sees half a file and a failed download never damages the last good copy.
    Raises on any failure; callers decide whether that is fatal.
    """
    path, meta_path = _paths(kind)
    meta = _read_meta(meta_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(IBTRACS_URL + FILES[kind])
    if path.exists() and meta.get("last_modified"):
        request.add_header("If-Modified-Since", meta["last_modified"])

    try:
        response = _open_url(request, DOWNLOAD_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        if exc.code == 304:  # unchanged upstream: just record that we checked
            _write_json_atomic(meta_path, {**meta, "synced_at": _now_iso()})
            return
        raise

    with response:
        last_modified = response.headers.get("Last-Modified")
        handle = tempfile.NamedTemporaryFile("wb", dir=CACHE_DIR, prefix=".tmp-", delete=False)
        try:
            with handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
            with open(handle.name, "rb") as downloaded:
                if downloaded.read(4) != b"SID,":
                    raise ValueError(f"{FILES[kind]} did not download as an IBTrACS CSV")
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    _write_json_atomic(meta_path, {"last_modified": last_modified, "synced_at": _now_iso()})


def _refresh_in_background(kind: str) -> None:
    with _refresh_guard:
        if kind in _refreshing:
            return
        _refreshing.add(kind)

    def run() -> None:
        try:
            with _download_locks[kind]:
                _download(kind)
        except Exception as exc:  # noqa: BLE001 - keep serving the last good copy
            logger.warning("IBTrACS refresh of %s failed, still serving the cached copy: %s", FILES[kind], exc)
        finally:
            with _refresh_guard:
                _refreshing.discard(kind)

    threading.Thread(target=run, name=f"ibtracs-refresh-{kind}", daemon=True).start()


def _ensure(kind: str) -> dict:
    """The cached file for `kind`, downloading it if there is none yet."""
    path, meta_path = _paths(kind)
    if not path.exists():
        with _download_locks[kind]:
            if not path.exists():
                try:
                    _download(kind)
                except Exception as exc:  # noqa: BLE001 - reported to the caller
                    raise CycloneDataUnavailableError(
                        f"could not download {FILES[kind]} from NOAA IBTrACS "
                        f"({type(exc).__name__}: {exc}), and there is no cached copy. "
                        "No cyclone track is shown rather than an invented one."
                    ) from exc

    meta = _read_meta(meta_path)
    synced = meta.get("synced_at")
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(synced.replace("Z", "+00:00"))).total_seconds()
    except (AttributeError, ValueError):
        age = math.inf
    stale = age > REFRESH_AFTER_SECONDS[kind]
    if stale:
        _refresh_in_background(kind)
    return {"path": path, "last_modified": meta.get("last_modified"), "synced_at": synced, "stale": stale}


# --------------------------------------------------------------------------
# Parsing and selection
# --------------------------------------------------------------------------


def _load_frame(path: Path) -> pd.DataFrame:
    """North Indian rows of one IBTrACS CSV, parsed once per file version."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _parsed_lock:
        if key in _parsed:
            return _parsed[key]

    header = pd.read_csv(path, nrows=0).columns
    frame = pd.read_csv(
        path,
        skiprows=[1],  # the second line holds units, not data
        usecols=[column for column in _COLUMNS if column in header],
        low_memory=False,
        keep_default_na=False,
        na_values=[" ", ""],
    )
    frame = frame[frame["BASIN"] == BASIN].copy()
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    for column in ("LAT", "LON", "WMO_WIND", "WMO_PRES"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ISO_TIME", "LAT", "LON"])

    with _parsed_lock:
        if len(_parsed) >= 8:
            _parsed.clear()
        _parsed[key] = frame
    return frame


def _grade(value) -> str | None:
    """IMD grade in canonical spelling, the raw string if unrecognised, or None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return _GRADE_BY_UPPER.get(text.upper(), text) if text else None


def _number(value, digits: int = 1) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _iso(moment) -> str | None:
    return None if pd.isna(moment) else moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _crosses_map(rows: pd.DataFrame) -> bool:
    min_lon, min_lat, max_lon, max_lat = MAP_BBOX
    return bool((rows["LON"].between(min_lon, max_lon) & rows["LAT"].between(min_lat, max_lat)).any())


def _reached_cyclonic_storm(rows: pd.DataFrame) -> bool:
    """IMD grade CS or stronger at any point.

    Falls back to IMD's wind only when a storm carries no grade IMD's scale
    recognises, so a spelling IBTrACS might use for a grade cannot silently drop
    the strongest storms.
    """
    grades = [_grade(value) for value in rows.get("NEWDELHI_GRADE", pd.Series(dtype=object)).dropna()]
    known = [grade for grade in grades if grade in IMD_GRADES]
    if known:
        return max(IMD_GRADES.index(grade) for grade in known) >= IMD_GRADES.index("CS")
    peak = rows["WMO_WIND"].max()
    return bool(pd.notna(peak) and peak >= CYCLONIC_STORM_KT)


def _storm(rows: pd.DataFrame, active: bool) -> dict:
    rows = rows.sort_values("ISO_TIME")
    winds = rows["WMO_WIND"]
    has_wind = bool(winds.notna().any())
    peak_row = rows.loc[winds.idxmax()] if has_wind else rows.iloc[-1]

    grades = [grade for grade in (_grade(v) for v in rows.get("NEWDELHI_GRADE", [])) if grade]
    known = [grade for grade in grades if grade in IMD_GRADES]
    peak_grade = max(known, key=IMD_GRADES.index) if known else (grades[-1] if grades else None)
    latest_grade = grades[-1] if grades else None
    latest_winds = winds.dropna()
    subbasin = str(rows["SUBBASIN"].iloc[0]) if "SUBBASIN" in rows else ""
    track_type = str(rows["TRACK_TYPE"].iloc[0]) if "TRACK_TYPE" in rows else ""

    return {
        "sid": str(rows["SID"].iloc[0]),
        "name": str(rows["NAME"].iloc[0]),
        "subbasin": SUBBASIN_NAMES.get(subbasin),
        "start": _iso(rows["ISO_TIME"].iloc[0]),
        "end": _iso(rows["ISO_TIME"].iloc[-1]),
        "active": active,
        "provisional": "PROVISIONAL" in track_type.upper(),
        "peak_wind_kt": _number(winds.max()) if has_wind else None,
        "min_pressure_hpa": _number(rows["WMO_PRES"].min()),
        "peak_grade": peak_grade,
        "peak_grade_name": IMD_GRADE_NAMES.get(peak_grade),
        "peak_lat": _number(peak_row["LAT"], 2),
        "peak_lon": _number(peak_row["LON"], 2),
        "latest_wind_kt": _number(latest_winds.iloc[-1]) if len(latest_winds) else None,
        "latest_grade": latest_grade,
        "latest_grade_name": IMD_GRADE_NAMES.get(latest_grade),
        "points": [
            {
                "time": _iso(row.ISO_TIME),
                "lat": round(float(row.LAT), 2),
                "lon": round(float(row.LON), 2),
                "wind_kt": _number(row.WMO_WIND),
                "pressure_hpa": _number(row.WMO_PRES),
                "grade": _grade(getattr(row, "NEWDELHI_GRADE", None)),
            }
            for row in rows.itertuples(index=False)
        ],
    }


def select_tracks(
    recent: pd.DataFrame, active: pd.DataFrame, *, days: int, now: datetime
) -> tuple[list[dict], list[dict]]:
    """(active storms, past storms), newest first. Pure: no files, no network."""
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    cutoff = pd.Timestamp(now) - pd.Timedelta(days=days)

    active_sids = set(active["SID"])
    active_storms = [_storm(rows, True) for _, rows in active.groupby("SID")]

    past_storms = []
    for sid, rows in recent.groupby("SID"):
        if sid in active_sids or rows["ISO_TIME"].max() < cutoff:
            continue
        if _crosses_map(rows) and _reached_cyclonic_storm(rows):
            past_storms.append(_storm(rows, False))

    newest_first = lambda storm: storm["end"]  # noqa: E731
    return sorted(active_storms, key=newest_first, reverse=True), sorted(past_storms, key=newest_first, reverse=True)


def get_tracks(days: int = DEFAULT_WINDOW_DAYS, now: datetime | None = None) -> dict:
    """Active and recent North Indian cyclone tracks, with full provenance."""
    if not 1 <= days <= MAX_WINDOW_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_WINDOW_DAYS}, got {days}")
    now = now or datetime.now(timezone.utc)

    recent_file = _ensure("recent")
    active_file = _ensure("active")
    recent = _load_frame(recent_file["path"])
    active = _load_frame(active_file["path"])
    active_storms, past_storms = select_tracks(recent, active, days=days, now=now)

    synced = [moment for moment in (recent_file["synced_at"], active_file["synced_at"]) if moment]
    min_lon, min_lat, max_lon, max_lat = MAP_BBOX
    return {
        "source": SOURCE,
        "source_url": IBTRACS_URL,
        "basin": BASIN,
        "wind_note": WIND_NOTE,
        "selection": SELECTION_NOTE.format(min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat, days=days),
        "window_days": days,
        "archive_updated": recent_file["last_modified"],
        "last_synced": min(synced) if synced else None,
        "stale": bool(recent_file["stale"] or active_file["stale"]),
        "newest_fix": _iso(recent["ISO_TIME"].max()) if len(recent) else None,
        "active_count": len(active_storms),
        "active": active_storms,
        "recent": past_storms,
    }
