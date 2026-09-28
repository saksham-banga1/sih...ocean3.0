"""
Reading the marine-heatwave record and its precomputed climatology off disk.

app/services/heatwave.py holds the definition and does no I/O. This module
does the I/O and none of the definition, so the arithmetic can be tested
against records built in a test file rather than against whatever happens to
be downloaded.

Two things live under data/model_history/sst_baseline/:

  thetao_<box>_<year>.nc   the daily SST record, one file per year, written by
                           scripts/download_sst_baseline.py
  heatwave_climatology.nc  per-cell day-of-year mean and 90th percentile,
                           written once by scripts/build_heatwave_climatology.py

Both come from the GLORYS12V1 REANALYSIS. The Sep 2026 date slider, the
heatmaps and the exports are driven by the analysis-forecast product in
data/model/ instead: two different models, never mixed, and the reanalysis
ends where its own record ends rather than being extended to meet the other.

With no climatology on disk the caller gets HeatwaveUnavailableError. Nothing
here falls back to a shorter baseline, a coarser window or a nearby cell --
a marine heatwave is defined against a 30-year baseline, and a number computed
against anything else is not one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

from app.services.heatwave import Climatology, HeatwaveDataError
from app.services.netcdf_lock import NETCDF_LOCK

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR.parent / "data" / "model_history" / "sst_baseline"
CLIMATOLOGY_NAME = "heatwave_climatology.nc"

_lock = threading.Lock()


class HeatwaveUnavailableError(RuntimeError):
    """No usable record on disk -- say so, never substitute."""


@dataclass(frozen=True)
class RecordInfo:
    """What the loaded record actually covers, read back from the files."""

    first_day: date
    last_day: date
    days: int
    lat: np.ndarray
    lon: np.ndarray
    depth_m: float | None
    units: str
    baseline_first_year: int
    baseline_last_year: int
    window_days: int
    percentile: float
    dataset_id: str
    smoothing: str


def data_dir() -> Path:
    return DATA_DIR


def _year_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("thetao_*.nc"))


@lru_cache(maxsize=4)
def _open_climatology(directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the precomputed climatology once per process."""
    path = Path(directory) / CLIMATOLOGY_NAME
    if not path.exists():
        raise HeatwaveUnavailableError(
            f"no marine-heatwave climatology at {path}. Build it with "
            "scripts/download_sst_baseline.py then scripts/build_heatwave_climatology.py"
        )
    with NETCDF_LOCK, xr.open_dataset(path) as ds:
        mean = np.asarray(ds["sst_mean"].values, dtype="float32")
        p90 = np.asarray(ds["sst_p90"].values, dtype="float32")
        lat = np.asarray(ds["latitude"].values, dtype="float64")
        lon = np.asarray(ds["longitude"].values, dtype="float64")
        attrs = dict(ds.attrs)
    return mean, p90, lat, lon, attrs


@lru_cache(maxsize=4)
def _open_record(directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Dates, latitudes and longitudes of the SST record, without its values.

    The values are read per request for the cells asked for: the whole record
    is several gigabytes and holding it in memory would cost more than reading
    the column back each time.
    """
    files = _year_files(Path(directory))
    if not files:
        raise HeatwaveUnavailableError(
            f"no SST record in {directory}. Download it with scripts/download_sst_baseline.py")
    with NETCDF_LOCK, xr.open_mfdataset([str(p) for p in files], combine="by_coords") as ds:
        dates = np.asarray(ds["time"].values, dtype="datetime64[D]")
        lat = np.asarray(ds["latitude"].values, dtype="float64")
        lon = np.asarray(ds["longitude"].values, dtype="float64")
        attrs = {
            "units": str(ds["thetao"].attrs.get("units", "")),
            "depth_m": (float(ds["depth"].values[0])
                        if "depth" in ds.coords and ds["depth"].size else None),
        }
    return dates, lat, lon, attrs


def record_info(directory: Path | None = None) -> RecordInfo:
    """What is on disk, read from the files rather than assumed."""
    where = str(directory or DATA_DIR)
    with _lock:
        dates, lat, lon, attrs = _open_record(where)
        _, _, clim_lat, clim_lon, clim_attrs = _open_climatology(where)

    if clim_lat.size != lat.size or clim_lon.size != lon.size:
        raise HeatwaveUnavailableError(
            f"the climatology grid ({clim_lat.size}x{clim_lon.size}) does not match the "
            f"record grid ({lat.size}x{lon.size}) -- rebuild it for this record")

    return RecordInfo(
        first_day=dates.min().astype(object),
        last_day=dates.max().astype(object),
        days=int(dates.size),
        lat=lat,
        lon=lon,
        depth_m=attrs["depth_m"],
        units=attrs["units"],
        baseline_first_year=int(clim_attrs.get("baseline_first_year", 0)),
        baseline_last_year=int(clim_attrs.get("baseline_last_year", 0)),
        window_days=int(clim_attrs.get("climatology_window_days", 0)),
        percentile=float(clim_attrs.get("threshold_percentile", 0.0)),
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        smoothing=str(clim_attrs.get("smoothing", "")),
    )


def nearest_cell(lat: float, lon: float, directory: Path | None = None) -> tuple[int, int, float, float]:
    """Index of the grid cell holding a position, and the cell's own centre.

    The centre is returned so the caller can report the position the numbers
    are actually for, rather than the position that was asked about.
    """
    where = str(directory or DATA_DIR)
    with _lock:
        _, lats, lons, _ = _open_record(where)
    if not (lats.min() <= lat <= lats.max() and lons.min() <= lon <= lons.max()):
        raise HeatwaveDataError(
            f"({lat:.3f}, {lon:.3f}) is outside the record, which covers "
            f"{lats.min():.2f}..{lats.max():.2f} N, {lons.min():.2f}..{lons.max():.2f} E")
    i = int(np.abs(lats - lat).argmin())
    j = int(np.abs(lons - lon).argmin())
    return i, j, float(lats[i]), float(lons[j])


def climatology_at(i: int, j: int, directory: Path | None = None) -> Climatology:
    """The precomputed day-of-year curves for one cell."""
    where = str(directory or DATA_DIR)
    with _lock:
        mean, p90, _, _, attrs = _open_climatology(where)
    column_mean = np.asarray(mean[:, i, j], dtype="float64")
    column_p90 = np.asarray(p90[:, i, j], dtype="float64")
    if not np.isfinite(column_mean).any():
        raise HeatwaveDataError(
            "this cell has no climatology -- it is land, or was masked throughout the baseline")
    return Climatology(
        mean=column_mean,
        threshold=column_p90,
        baseline_first_year=int(attrs.get("baseline_first_year", 0)),
        baseline_last_year=int(attrs.get("baseline_last_year", 0)),
        window_days=int(attrs.get("climatology_window_days", 0)),
        percentile=float(attrs.get("threshold_percentile", 0.0)),
        smoothed_days=None,
    )


def series_at(i: int, j: int, directory: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The whole daily SST record for one cell."""
    where = str(directory or DATA_DIR)
    files = _year_files(Path(where))
    if not files:
        raise HeatwaveUnavailableError(f"no SST record in {where}")
    with NETCDF_LOCK, xr.open_mfdataset([str(p) for p in files], combine="by_coords") as ds:
        column = ds["thetao"].isel(latitude=i, longitude=j)
        if "depth" in column.dims:
            column = column.isel(depth=0)
        values = np.asarray(column.values, dtype="float64")
        dates = np.asarray(ds["time"].values, dtype="datetime64[D]")
    return dates, values


def field_on(day: date, directory: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SST, baseline mean and threshold across the whole grid for one day.

    Returned as three arrays of the same shape so the caller can classify every
    cell at once. A day outside the record raises rather than returning the
    nearest one: this module never quietly answers a different question.
    """
    where = str(directory or DATA_DIR)
    files = _year_files(Path(where))
    if not files:
        raise HeatwaveUnavailableError(f"no SST record in {where}")
    wanted = np.datetime64(day, "D")

    with _lock:
        mean, p90, _, _, _ = _open_climatology(where)

    year_file = [p for p in files if p.stem.rsplit("_", 1)[1] == str(day.year)]
    if not year_file:
        raise HeatwaveDataError(f"no SST file covering {day} in {where}")
    with NETCDF_LOCK, xr.open_dataset(year_file[0]) as ds:
        dates = np.asarray(ds["time"].values, dtype="datetime64[D]")
        hits = np.flatnonzero(dates == wanted)
        if hits.size == 0:
            raise HeatwaveDataError(
                f"{day} is not in the record, which for {day.year} runs "
                f"{str(dates.min())} .. {str(dates.max())}")
        field = ds["thetao"].isel(time=int(hits[0]))
        if "depth" in field.dims:
            field = field.isel(depth=0)
        sst = np.asarray(field.values, dtype="float32")

    doy = min(day.timetuple().tm_yday, 365)
    return sst, mean[doy - 1], p90[doy - 1]
