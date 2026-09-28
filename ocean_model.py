"""
Read numerical model NetCDF files from data/model/ and reshape them
into JSON-serializable grids for the frontend.

The frontend (main.js) talks in its own variable names -- "sst", "salinity",
"wave" -- while the Copernicus files use CF names (thetao, so). This module owns
that translation, so routers and main.js never need to know about CF naming or
the on-disk filename scheme produced by scripts/download_ocean_data.py.

Most fields are Copernicus numerical model output. Sea level anomaly ("sla") is
the exception: it is observed satellite altimetry, served through the same grid
path only because it has the same shape. NC_PROVENANCE records which is which.

Other fields are in no file at all: DERIVED_FIELDS calculates them from one that
is -- tropical cyclone heat potential from model temperature.
"""

from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

from app.services.heat_content import tropical_cyclone_heat_potential
from app.services.isotherms import d20_depth, d26_depth
from app.services.netcdf_lock import NETCDF_LOCK

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
# data/ lives at the project root, alongside backend/ and frontend/.
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "model"

# Frontend variable name -> NetCDF variable name written by download_ocean_data.py.
# uo/vo are the eastward/northward velocity components from the '-cur-' dataset
# of the same product family; names verified against copernicusmarine.describe().
VARIABLE_TO_NC = {
    "sst": "thetao",
    "salinity": "so",
    "uo": "uo",
    "vo": "vo",
    # 2D fields -- no vertical axis. mlotst is the model's mixed layer
    # thickness; sla is a SATELLITE product (DUACS altimetry), not model output.
    "mld": "mlotst",
    "sla": "sla",
    # Biogeochemistry: dissolved oxygen and nitrate, on a coarser 0.25 deg grid
    # than the physics fields but the same vertical levels.
    "o2": "o2",
    "no3": "no3",
    # Satellite ocean colour: surface only, and published under its CF name in
    # upper case. Gap-filled under cloud -- see the flags in the same file.
    "chl": "CHL",
    # The other two macronutrients, in the same file family as nitrate.
    "po4": "po4",
    "si": "si",
    # Model chlorophyll, WITH depth -- a different field from the satellite
    # surface product above. Note the trap: resolve_nc_variable() lower-cases
    # its argument, so the bare CF name "chl" resolves to the SATELLITE product
    # because "chl" is already a key here. Model chlorophyll must be asked for
    # by its own key, "chl_model", which is why it has one.
    "chl_model": "chl",
    # The model's own sea surface height above the geoid. 2D, and not the same
    # quantity as the satellite anomaly "sla": this one carries the mean
    # dynamic topography, so its values sit around a metre, not around zero.
    "zos": "zos",
    # Significant wave height, from the wave model. The only INSTANTANEOUS field
    # here: the product is 3-hourly and only the 12:00 UTC step is downloaded,
    # so this is a snapshot at midday, not a daily mean like the others.
    "wave": "VHM0",
    # Photosynthetically active radiation, WITH depth. DERIVED, not model
    # output: no model or satellite product here publishes PAR for these days,
    # so scripts/build_par.py calculates it from ECMWF surface sunlight and the
    # model chlorophyll profile (Morel et al. 2007) and writes it as a file of
    # its own, which is why it can be read like any other field.
    "par": "par",
}

# Fields with no depth dimension. A request's depth is ignored for these and the
# grid's actual_depth is None, because there is no level to report -- claiming
# 0 m would be false for mixed layer depth, whose values are themselves depths.
SURFACE_NC_VARIABLES = frozenset({"mlotst", "sla", "CHL", "zos", "VHM0"})

# The gap-free ocean-colour product fills cloud gaps by interpolation and records
# where in its "flags" variable, as bits: 1 land, 2 interpolated. In the monsoon
# that can be half a map, so the share is reported alongside the data rather than
# left for the viewer to assume.
CHL_FLAG_LAND = 1
CHL_FLAG_INTERPOLATED = 2

# Where each scalar field comes from, so a response can say so.
NC_PROVENANCE = {
    "thetao": ("model", "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"),
    "so": ("model", "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"),
    "mlotst": ("model", "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"),
    "sla": ("satellite", "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"),
    "o2": ("model", "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"),
    "no3": ("model", "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m"),
    "CHL": ("satellite", "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"),
    "po4": ("model", "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m"),
    "si": ("model", "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m"),
    "chl": ("model", "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m"),
    "zos": ("model", "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"),
    "VHM0": ("model", "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"),
    "par": ("derived", "ECMWF ERA5/IFS shortwave (Open-Meteo) + cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m chlorophyll"),
}

# Fields calculated from a downloaded variable rather than read from a file of
# their own: name -> (variable it is calculated from, calculation). Every one is
# 2D, summarising a whole water column. Kept out of VARIABLE_TO_NC on purpose:
# /api/model/dates reverses that mapping to label files, and "thetao" has to
# stay "sst" there.
DERIVED_FIELDS = {
    "tchp": ("thetao", tropical_cyclone_heat_potential),
    # How deep the cyclone-fuel layer reaches, and the thermocline marker. Both
    # are NaN wherever the isotherm does not exist -- never 0, which would claim
    # the crossing sits exactly at the sea surface.
    "d26": ("thetao", d26_depth),
    "d20": ("thetao", d20_depth),
}

# Currents are a vector pair, not a scalar field, so they get their own path
# through the API rather than a slot in the scalar grid endpoint.
CURRENT_COMPONENTS = ("uo", "vo")

# Fields rounded to significant figures rather than decimal places. PAR falls
# off exponentially with depth -- tens of mol/m2/day at the surface, hundredths
# below the euphotic zone -- and three decimals would flatten everything below
# about 80 m to 0.
SIGNIFICANT_FIGURES = {"par": 3}


def _round_value(value: float, round_to: int, sig_figs: int | None = None) -> float:
    if sig_figs:
        return float(f"{value:.{sig_figs}g}")
    return round(value, round_to)


# Units for each frontend variable, so callers can label values without guessing.
VARIABLE_UNITS = {
    "sst": "degrees_C",
    "salinity": "PSU",
    "mld": "m",
    "sla": "m",
    "tchp": "kJ/cm2",
    "d26": "m",
    "d20": "m",
    "wave": "m",
    "o2": "mmol/m3",
    "no3": "mmol/m3",
    "chl": "mg/m3",
    "po4": "mmol/m3",
    "si": "mmol/m3",
    "chl_model": "mg/m3",
    "zos": "m",
    "par": "mol/m2/day",
}

# thetao_bay_of_bengal_20260922-20260922_0-2000m.nc
# thetao_80E-92E_8N-20N_20260922-20260922_0-2000m.nc
# The region label may itself contain '_' and '-', so anchor on the date range:
# the greedy region group backtracks to the last YYYYMMDD-YYYYMMDD it can find.
FILENAME_RE = re.compile(
    # A CF name may carry digits after its first letter ("o2", "no3") and satellite
    # products publish theirs in upper case ("CHL"), so a lower-case, letters-only
    # pattern would make those files invisible to everything here.
    r"^(?P<nc_var>[A-Za-z][A-Za-z0-9]*)_(?P<region>.+)_"
    r"(?P<start>\d{8})-(?P<end>\d{8})_"
    r"(?P<min_depth>[\d.]+)-(?P<max_depth>[\d.]+)m\.nc$"
)

# Datasets are loaded fully into memory, so keep only a few. Each Bay of Bengal
# day is ~8 MB on disk.
CACHE_MAX_ENTRIES = 2

_cache: OrderedDict[tuple[tuple[Path, ...], date], xr.Dataset] = OrderedDict()
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0

# Flattened grids, keyed by the exact request. Reading the NetCDF is already
# cached above, but re-flattening a depth slice costs ~8 ms per request and the
# frontend asks for the same slice repeatedly (re-render, revisit, reload).
# Entries are small (a Bay of Bengal slice at stride 2 is ~16k dicts).
GRID_CACHE_MAX_ENTRIES = 16

_grid_cache: OrderedDict[tuple, list[dict]] = OrderedDict()
_grid_cache_lock = threading.Lock()
_grid_hits = 0
_grid_misses = 0


class DatasetNotFoundError(FileNotFoundError):
    """No downloaded file covers the requested variable and date."""


def resolve_nc_variable(variable: str) -> str:
    """Map a frontend variable name ('sst') to the NetCDF variable it is read from
    ('thetao'). A derived field resolves to the variable it is calculated from,
    which is what file lookup and date coverage need."""
    key = variable.lower()
    if key in VARIABLE_TO_NC:
        return VARIABLE_TO_NC[key]
    if key in DERIVED_FIELDS:
        return DERIVED_FIELDS[key][0]
    # Also accept the CF name directly, so callers can pass 'thetao' if they prefer.
    if key in VARIABLE_TO_NC.values():
        return key
    raise ValueError(
        f"unknown variable {variable!r} -- expected one of "
        f"{sorted(VARIABLE_TO_NC)} or {sorted(VARIABLE_TO_NC.values())}"
    )


def is_surface_variable(variable: str) -> bool:
    """True for 2D fields with no depth axis: mixed layer depth, sea level anomaly,
    and every derived field, each of which summarises a whole water column."""
    if variable.lower() in DERIVED_FIELDS:
        return True
    return resolve_nc_variable(variable) in SURFACE_NC_VARIABLES


def provenance(variable: str) -> tuple[str, str | None]:
    """(source, dataset_id) for a scalar field. source is 'model', 'satellite',
    or 'derived' for a quantity calculated from model output."""
    key = variable.lower()
    if key in DERIVED_FIELDS:
        base = DERIVED_FIELDS[key][0]
        return "derived", NC_PROVENANCE.get(base, (None, None))[1]
    return NC_PROVENANCE.get(resolve_nc_variable(key), ("model", None))


def _depth_dim(ds: xr.Dataset | xr.DataArray) -> str | None:
    """The vertical dimension, or None for a 2D field."""
    for name in ("depth", "elevation"):
        if name in ds.dims:
            return name
    return None


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"invalid date {value!r} -- expected YYYY-MM-DD")


# The days a file really holds, read from its time axis and cached per file
# version. A file is named for the range that was ASKED for; a product that lags
# (satellite chlorophyll runs about two days behind) comes back shorter, and a
# day it lacks must be reported as a fallback, not served as if it were there.
_file_days: dict[tuple[str, float], tuple[date, ...] | None] = {}
_file_days_lock = threading.Lock()


def _days_in_file(path: Path) -> tuple[date, ...] | None:
    key = (str(path), path.stat().st_mtime)
    with _file_days_lock:
        if key in _file_days:
            return _file_days[key]
    days = None
    try:
        with NETCDF_LOCK, xr.open_dataset(path) as ds:
            if "time" in ds.coords and ds["time"].size:
                stamps = np.atleast_1d(ds["time"].values).astype("datetime64[D]")
                days = tuple(sorted({datetime.strptime(str(d), "%Y-%m-%d").date() for d in stamps}))
    except Exception:  # noqa: BLE001 -- an unreadable time axis falls back to the file name
        days = None
    with _file_days_lock:
        _file_days[key] = days
    return days


def iter_available_files(data_dir: Path | None = None) -> Iterator[dict]:
    """Yield metadata for every recognised .nc file in the model data directory.
    `days` are the days the file really holds; `start`/`end` bound them."""
    directory = data_dir or DATA_DIR
    if not directory.is_dir():
        return

    for path in sorted(directory.glob("*.nc")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        parts = match.groupdict()
        named_start = datetime.strptime(parts["start"], "%Y%m%d").date()
        named_end = datetime.strptime(parts["end"], "%Y%m%d").date()
        days = _days_in_file(path)
        if not days:
            days = tuple(named_start + timedelta(days=k) for k in range((named_end - named_start).days + 1))
        yield {
            "path": path,
            "nc_var": parts["nc_var"],
            "region": parts["region"],
            "start": days[0],
            "end": days[-1],
            "days": days,
            "min_depth": float(parts["min_depth"]),
            "max_depth": float(parts["max_depth"]),
        }


def find_dataset_paths(
    variable: str, date_str: str | date, data_dir: Path | None = None
) -> list[Path]:
    """Every file for this variable covering this date, one per downloaded region.

    Regions are downloaded separately (arabian_sea, bay_of_bengal, ...), so a
    request spanning the whole basin needs all of them. Returning only one would
    silently drop half the map.
    """
    nc_var = resolve_nc_variable(variable)
    wanted = _parse_date(date_str)

    candidates = [
        entry
        for entry in iter_available_files(data_dir)
        if entry["nc_var"] == nc_var and wanted in entry["days"]
    ]

    if not candidates:
        directory = data_dir or DATA_DIR
        available = [
            f"{e['nc_var']} {e['region']} {e['start']}..{e['end']}"
            for e in iter_available_files(data_dir)
        ]
        raise DatasetNotFoundError(
            f"no downloaded file covers variable {variable!r} ({nc_var}) on {wanted}.\n"
            f"  Looked in: {directory}\n"
            f"  Available: {available or 'nothing'}\n"
            f"  Download it with: python scripts/download_ocean_data.py "
            f"--variables {nc_var} --start-date {wanted} --end-date {wanted}"
        )

    # One file per region. Where two files cover the same region and date, the
    # narrowest date range wins, then the most recently modified.
    candidates.sort(
        key=lambda e: ((e["end"] - e["start"]).days, -e["path"].stat().st_mtime)
    )
    by_region: dict[str, Path] = {}
    for entry in candidates:
        by_region.setdefault(entry["region"], entry["path"])

    return sorted(by_region.values())


def load_dataset(
    variable: str, date: str, data_dir: Path | None = None
) -> xr.Dataset:
    """Load the model file for a frontend variable ('sst'/'salinity') and date.

    Datasets are cached in-process and keyed by resolved file path, so repeat
    requests for the same date -- or for two dates inside one file's range --
    reuse a single in-memory copy instead of re-reading from disk.

    A file downloaded over a date range holds one timestep per day, so the
    requested day is selected before returning. Without that, every date inside
    a multi-day file would silently return the first day's data.
    """
    global _cache_hits, _cache_misses

    paths = find_dataset_paths(variable, date, data_dir)
    wanted = _parse_date(date)
    # Keyed by date as well as paths, because only the requested day is loaded.
    key = (tuple(paths), wanted)

    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)  # most-recently-used last
            _cache_hits += 1
            return _cache[key]

    # Read outside the lock: decoding is slow and shouldn't block other readers.
    # Each file is narrowed to the requested day before loading, so a six-day
    # download costs one timestep in memory rather than six.
    parts = []
    with NETCDF_LOCK:        # netCDF-C is not thread-safe; see netcdf_lock.py
        for path in paths:
            handle = xr.open_dataset(path)
            try:
                parts.append(_select_day(handle, wanted).load())
            finally:
                handle.close()

    if len(parts) == 1:
        dataset = parts[0]
    else:
        # Regions tile west-to-east and share their boundary longitude, so
        # concatenate in longitude order and drop the duplicated column.
        parts.sort(key=lambda part: float(part.longitude.min()))
        dataset = xr.concat(parts, dim="longitude")
        _, first_occurrence = np.unique(dataset.longitude.values, return_index=True)
        dataset = dataset.isel(longitude=np.sort(first_occurrence))

    with _cache_lock:
        if key in _cache:
            # Another thread won the race; keep its copy and discard ours.
            dataset.close()
            _cache.move_to_end(key)
            _cache_hits += 1
            return _cache[key]

        _cache[key] = dataset
        _cache_misses += 1
        while len(_cache) > CACHE_MAX_ENTRIES:
            _, evicted = _cache.popitem(last=False)  # drop least-recently-used
            evicted.close()

    return dataset


def _select_day(dataset: xr.Dataset, wanted: date) -> xr.Dataset:
    """Narrow a multi-day file to the requested day's timestep.

    Most products here are daily means with one step per day. The wave model is
    instantaneous and published every three hours, so its files hold eight steps
    for the day -- and a bare date resolves to midnight, which would quietly
    serve the 00:00 field while the rest of the app talks about midday.

    So: take the steps that fall inside the day, and if there is more than one,
    the one nearest DAILY_MEAN_CENTRE_HOUR. Slicing to the day first matters --
    asking for "nearest to midday" across the whole file would sit exactly 12 h
    from two different daily means and could tie-break onto the wrong day.
    """
    if "time" not in dataset.dims or dataset.sizes.get("time", 1) <= 1:
        return dataset

    day = dataset.sel(time=slice(f"{wanted}T00:00:00", f"{wanted}T23:59:59.999999"))
    steps = day.sizes.get("time", 0)
    if steps == 0:
        # Nothing inside the day: fall back to the nearest step in the file, as
        # before, so a gap still serves its closest neighbour.
        return dataset.sel(time=str(wanted), method="nearest").expand_dims("time")
    if steps == 1:
        return day

    target = np.datetime64(f"{wanted}T{DAILY_MEAN_CENTRE_HOUR:02d}:00:00")
    nearest = int(np.abs(day["time"].values - target).argmin())
    # Keep the time dimension so downstream isel(time=0) still works.
    return day.isel(time=[nearest])


def _coord_name(ds: xr.Dataset | xr.DataArray, *candidates: str) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"none of {candidates} found in dataset coordinates {list(ds.coords)}")


def get_grid_at_depth(
    ds: xr.Dataset,
    depth: float,
    variable: str,
    stride: int = 1,
    round_to: int = 3,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Flatten one depth level into [{lat, lon, value}, ...], land removed.

    Selects the nearest available depth level, takes the first timestep, drops
    NaN cells (land), and returns plain Python floats so the result drops
    straight into a JSON response.

    stride > 1 subsamples every Nth cell in each direction -- the full Bay of
    Bengal grid is 241x217, which is ~36k ocean points once land is removed.

    bbox is (min_lon, min_lat, max_lon, max_lat), the same west/south/east/north
    order Cesium's Rectangle.fromDegrees uses in main.js. Clipping here rather
    than filtering the flat list keeps the work inside xarray.
    """
    # Derived fields are calculated from their base variable, not read from it.
    if variable.lower() in DERIVED_FIELDS:
        return get_derived_grid(ds, variable, stride=stride, round_to=round_to, bbox=bbox)

    nc_var = resolve_nc_variable(variable)
    if nc_var not in ds:
        raise KeyError(
            f"variable {nc_var!r} not in dataset (has {list(ds.data_vars)})"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    da = ds[nc_var]

    if "time" in da.dims:
        da = da.isel(time=0)

    # A 2D field (mixed layer depth, sea level anomaly) has no vertical axis, so
    # there is no level to select and the requested depth is simply ignored.
    depth_name = _depth_dim(da)
    if depth_name is not None:
        da = da.sel({depth_name: depth}, method="nearest")

    region = _select_region(da, bbox, stride)
    return [] if region is None else _flatten_field(region, round_to, SIGNIFICANT_FIGURES.get(variable.lower()))


def _select_region(
    da: xr.DataArray,
    bbox: tuple[float, float, float, float] | None,
    stride: int,
) -> xr.DataArray | None:
    """Clip to bbox and thin to every Nth cell; None if the box holds no cells."""
    lat_name = _coord_name(da, "latitude", "lat")
    lon_name = _coord_name(da, "longitude", "lon")

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        # Coordinates are ascending in the Copernicus grids, so a plain slice works.
        da = da.sel(
            {
                lat_name: slice(min_lat, max_lat),
                lon_name: slice(min_lon, max_lon),
            }
        )
        if da[lat_name].size == 0 or da[lon_name].size == 0:
            return None

    if stride > 1:
        da = da.isel({lat_name: slice(None, None, stride), lon_name: slice(None, None, stride)})
    return da


def _flatten_field(da: xr.DataArray, round_to: int, sig_figs: int | None = None) -> list[dict]:
    """A 2D (lat, lon) field as [{lat, lon, value}, ...], NaN (land) dropped."""
    lat_name = _coord_name(da, "latitude", "lat")
    lon_name = _coord_name(da, "longitude", "lon")
    lats = da[lat_name].values
    lons = da[lon_name].values
    values = da.transpose(lat_name, lon_name).values

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    ocean = ~np.isnan(values)  # NaN marks land in the Copernicus grids

    return [
        {
            "lat": round(float(lat), round_to),
            "lon": round(float(lon), round_to),
            "value": _round_value(float(value), round_to, sig_figs),
        }
        for lat, lon, value in zip(
            lat_grid[ocean], lon_grid[ocean], values[ocean]
        )
    ]


def interpolated_fraction(
    variable: str,
    date_str: str,
    *,
    stride: int = 1,
    bbox: tuple[float, float, float, float] | None = None,
    data_dir: Path | None = None,
) -> float | None:
    """Share of the water cells in this view that were interpolated, not observed.

    None for any product without gap-fill flags. Measured over exactly the cells
    the grid returns, so it describes the map on screen rather than the archive.
    """
    if resolve_nc_variable(variable) != "CHL":
        return None

    dataset = load_dataset(variable, date_str, data_dir)
    if "flags" not in dataset:
        return None

    flags = dataset["flags"]
    if "time" in flags.dims:
        flags = flags.isel(time=0)
    region = _select_region(flags, bbox, stride)
    if region is None:
        return None

    values = np.asarray(region.values)
    finite = np.isfinite(values)
    codes = np.where(finite, values, 0).astype(np.int16)
    water = finite & ((codes & CHL_FLAG_LAND) == 0)
    if not water.any():
        return None
    interpolated = water & ((codes & CHL_FLAG_INTERPOLATED) != 0)
    return round(float(interpolated.sum() / water.sum()), 4)


def get_derived_grid(
    ds: xr.Dataset,
    variable: str,
    stride: int = 1,
    round_to: int = 3,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Flatten a derived 2D field (see DERIVED_FIELDS) into [{lat, lon, value}, ...].

    The region is clipped and thinned before calculating, not after: each
    column's value depends only on that column, so the returned cells are the
    same either way and there is far less to compute.
    """
    base, calculate = DERIVED_FIELDS[variable.lower()]
    if base not in ds:
        raise KeyError(f"variable {base!r} not in dataset (has {list(ds.data_vars)})")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    da = ds[base]
    if "time" in da.dims:
        da = da.isel(time=0)

    region = _select_region(da, bbox, stride)
    return [] if region is None else _flatten_field(calculate(region), round_to)


def available_days(variable: str, data_dir: Path | None = None) -> list[date]:
    """Every individual day this variable has downloaded data for, sorted."""
    nc_var = resolve_nc_variable(variable)
    days: set[date] = set()
    for entry in iter_available_files(data_dir):
        if entry["nc_var"] != nc_var:
            continue
        days.update(entry["days"])
    return sorted(days)


def find_nearest_available_date(
    variable: str, date_str: str | date, data_dir: Path | None = None
) -> date | None:
    """Closest downloaded day to the one asked for, or None if nothing exists.

    Lets a caller degrade to "showing nearest available" instead of a blank map
    when a date was never downloaded. Ties go to the earlier day.
    """
    wanted = _parse_date(date_str)
    days = available_days(variable, data_dir)
    if not days:
        return None
    return min(days, key=lambda day: (abs((day - wanted).days), day))


def get_grid(
    variable: str,
    date_str: str,
    depth: float,
    *,
    stride: int = 1,
    bbox: tuple[float, float, float, float] | None = None,
    round_to: int = 3,
    data_dir: Path | None = None,
) -> tuple[list[dict], float]:
    """Cached (points, actual_depth) for one depth slice.

    Keyed by the resolved file paths as well as the request, so downloading new
    data invalidates the entry naturally rather than serving a stale grid.
    """
    global _grid_hits, _grid_misses

    paths = find_dataset_paths(variable, date_str, data_dir)  # raises if nothing covers it
    nc_var = resolve_nc_variable(variable)
    key = (
        tuple(paths),
        str(_parse_date(date_str)),
        # Derived fields share their base variable's files, so key them by name.
        variable.lower() if variable.lower() in DERIVED_FIELDS else nc_var,
        # A 2D field is identical at every requested depth, so one entry serves all.
        None if is_surface_variable(variable) else float(depth),
        int(stride),
        bbox,
        int(round_to),
    )

    with _grid_cache_lock:
        if key in _grid_cache:
            _grid_cache.move_to_end(key)
            _grid_hits += 1
            return _grid_cache[key]

    dataset = load_dataset(variable, date_str, data_dir)
    points = get_grid_at_depth(
        dataset, depth, variable, stride=stride, round_to=round_to, bbox=bbox
    )
    # A 2D or derived field has no level to report, even when its base file has.
    actual_depth = None if is_surface_variable(variable) else get_selected_depth(dataset, depth)
    result = (points, actual_depth)

    with _grid_cache_lock:
        _grid_cache[key] = result
        _grid_misses += 1
        while len(_grid_cache) > GRID_CACHE_MAX_ENTRIES:
            _grid_cache.popitem(last=False)  # drop least-recently-used

    return result


def get_current_vectors(
    date_str: str,
    depth: float,
    *,
    stride: int = 8,
    bbox: tuple[float, float, float, float] | None = None,
    round_to: int = 4,
    data_dir: Path | None = None,
) -> tuple[list[dict], float]:
    """Downsampled current vectors as [{lat, lon, u, v, speed}, ...].

    u/v are the model's own eastward/northward velocities in m/s at the nearest
    depth level -- nothing is derived or invented. speed = sqrt(u^2 + v^2).

    Both components are read through the same load_dataset() path as every other
    variable, so regional merging, per-day selection and the dataset cache all
    apply unchanged. Cells where either component is land (NaN) are dropped, so
    a vector is only emitted where the model has a genuine velocity pair.
    """
    global _grid_hits, _grid_misses

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    # Cache on the resolved files of BOTH components, so a new download of
    # either one invalidates the entry.
    paths = tuple(
        path
        for component in CURRENT_COMPONENTS
        for path in find_dataset_paths(component, date_str, data_dir)
    )
    key = ("currents", paths, str(_parse_date(date_str)), float(depth),
           int(stride), bbox, int(round_to))

    with _grid_cache_lock:
        if key in _grid_cache:
            _grid_cache.move_to_end(key)
            _grid_hits += 1
            return _grid_cache[key]

    slices: dict[str, np.ndarray] = {}
    coords: dict[str, np.ndarray] = {}
    actual_depth = float(depth)

    for component in CURRENT_COMPONENTS:
        dataset = load_dataset(component, date_str, data_dir)
        da = dataset[resolve_nc_variable(component)]
        if "time" in da.dims:
            da = da.isel(time=0)

        depth_name = _coord_name(da, "depth", "elevation")
        da = da.sel({depth_name: depth}, method="nearest")
        actual_depth = get_selected_depth(dataset, depth)

        lat_name = _coord_name(da, "latitude", "lat")
        lon_name = _coord_name(da, "longitude", "lon")

        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            da = da.sel(
                {lat_name: slice(min_lat, max_lat), lon_name: slice(min_lon, max_lon)}
            )
            if da[lat_name].size == 0 or da[lon_name].size == 0:
                return [], actual_depth

        # Downsample on the grid itself, before flattening: taking every Nth
        # cell keeps the arrows evenly spaced geographically and keeps the
        # payload small, instead of thinning a huge list afterwards.
        da = da.isel(
            {lat_name: slice(None, None, stride), lon_name: slice(None, None, stride)}
        )

        slices[component] = da.transpose(lat_name, lon_name).values
        coords["lat"] = da[lat_name].values
        coords["lon"] = da[lon_name].values

    u = slices["uo"]
    v = slices["vo"]

    lat_grid, lon_grid = np.meshgrid(coords["lat"], coords["lon"], indexing="ij")
    # Both components must be present: a half-known vector is not a vector.
    ocean = np.isfinite(u) & np.isfinite(v)

    speed = np.sqrt(u**2 + v**2)

    vectors = [
        {
            "lat": round(float(la), 3),
            "lon": round(float(lo), 3),
            "u": round(float(uu), round_to),
            "v": round(float(vv), round_to),
            "speed": round(float(sp), round_to),
        }
        for la, lo, uu, vv, sp in zip(
            lat_grid[ocean], lon_grid[ocean], u[ocean], v[ocean], speed[ocean]
        )
    ]

    result = (vectors, actual_depth)
    with _grid_cache_lock:
        _grid_cache[key] = result
        _grid_misses += 1
        while len(_grid_cache) > GRID_CACHE_MAX_ENTRIES:
            _grid_cache.popitem(last=False)

    return result


def sample_at_point(
    ds: xr.Dataset, lat: float, lon: float, depth: float, variable: str
) -> float | None:
    """Model value at the grid cell nearest one position, or None over land.

    Used to give an observation its matching model value, which is the whole
    point of the model-vs-observation comparison in the frontend.
    """
    nc_var = resolve_nc_variable(variable)
    if nc_var not in ds:
        return None

    da = ds[nc_var]
    if "time" in da.dims:
        da = da.isel(time=0)

    depth_name = _coord_name(da, "depth", "elevation")
    lat_name = _coord_name(da, "latitude", "lat")
    lon_name = _coord_name(da, "longitude", "lon")

    # Outside the downloaded box, 'nearest' would silently snap to an edge cell,
    # so refuse rather than return a value from somewhere else.
    lats = da[lat_name].values
    lons = da[lon_name].values
    if not (lats.min() <= lat <= lats.max() and lons.min() <= lon <= lons.max()):
        return None

    value = da.sel(
        {depth_name: depth, lat_name: lat, lon_name: lon}, method="nearest"
    ).values

    return None if np.isnan(value) else _round_value(float(value), 3, SIGNIFICANT_FIGURES.get(variable.lower()))


# These files are daily MEANS stamped at 00:00 UTC with no time_bnds, so each
# stamp labels a whole calendar day. The representative instant of that average
# is midday, and no daily mean can resolve time better than +/-12 h anyway.
DAILY_MEAN_CENTRE_HOUR = 12

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def available_timesteps(variable: str, data_dir: Path | None = None) -> list[datetime]:
    """Representative instant of every downloaded daily mean, sorted."""
    return [
        datetime(day.year, day.month, day.day, DAILY_MEAN_CENTRE_HOUR)
        for day in available_days(variable, data_dir)
    ]


def find_nearest_timestep(
    variable: str, target: datetime, data_dir: Path | None = None
) -> tuple[date, datetime, float] | None:
    """Closest downloaded daily mean to an observation time.

    Returns (model_day, representative_instant, separation_hours). This is what
    makes the comparison a collocation rather than "whatever date the UI has
    selected" -- the observation's own timestamp picks the timestep.
    """
    steps = available_timesteps(variable, data_dir)
    if not steps:
        return None
    nearest = min(steps, key=lambda step: abs((step - target).total_seconds()))
    hours = abs((nearest - target).total_seconds()) / 3600.0
    return nearest.date(), nearest, round(hours, 2)


def find_nearest_valid_cell(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    variable: str,
    *,
    search_degrees: tuple[float, ...] = (0.5, 1.5, 3.0),
) -> tuple[float, float, float] | None:
    """Nearest model cell that actually holds water, as (lat, lon, distance_km).

    Plain .sel(method='nearest') can land on a land cell and yield nothing, which
    silently drops floats near a coast. This widens the search until it finds a
    wet column, so the caller gets a real neighbour plus the distance to it.
    """
    nc_var = resolve_nc_variable(variable)
    if nc_var not in ds:
        return None

    da = ds[nc_var]
    if "time" in da.dims:
        da = da.isel(time=0)

    depth_name = _coord_name(da, "depth", "elevation")
    lat_name = _coord_name(da, "latitude", "lat")
    lon_name = _coord_name(da, "longitude", "lon")

    # Wet where any level in the column is finite.
    wet = np.isfinite(da).any(dim=depth_name)

    for radius in search_degrees:
        window = wet.sel(
            {
                lat_name: slice(lat - radius, lat + radius),
                lon_name: slice(lon - radius, lon + radius),
            }
        )
        if window[lat_name].size == 0 or window[lon_name].size == 0:
            continue

        mask = window.values
        if not mask.any():
            continue

        lats = window[lat_name].values
        lons = window[lon_name].values
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        candidate_lats = lat_grid[mask]
        candidate_lons = lon_grid[mask]

        # Equirectangular ranking is fine at this scale; the winner is then
        # measured properly with haversine.
        dlat = candidate_lats - lat
        dlon = (candidate_lons - lon) * math.cos(math.radians(lat))
        best = int(np.argmin(dlat**2 + dlon**2))

        best_lat = float(candidate_lats[best])
        best_lon = float(candidate_lons[best])
        return best_lat, best_lon, round(haversine_km(lat, lon, best_lat, best_lon), 3)

    return None


def sample_profile_at_point(
    ds: xr.Dataset, lat: float, lon: float, variable: str
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Full model water column at the nearest grid cell.

    Returns (depths, values, grid_lat, grid_lon) with land/NaN levels dropped,
    or None if the position is outside the downloaded box or the column is all
    land. The grid coordinates are returned so the caller can report how far the
    nearest cell sits from the observation.
    """
    nc_var = resolve_nc_variable(variable)
    if nc_var not in ds:
        return None

    da = ds[nc_var]
    if "time" in da.dims:
        da = da.isel(time=0)

    depth_name = _coord_name(da, "depth", "elevation")
    lat_name = _coord_name(da, "latitude", "lat")
    lon_name = _coord_name(da, "longitude", "lon")

    # Same guard as sample_at_point: outside the box, 'nearest' would silently
    # snap to an edge cell and return values from somewhere else entirely.
    lats = da[lat_name].values
    lons = da[lon_name].values
    if not (lats.min() <= lat <= lats.max() and lons.min() <= lon <= lons.max()):
        return None

    column = da.sel({lat_name: lat, lon_name: lon}, method="nearest")

    depths = column[depth_name].values.astype(float)
    values = column.values.astype(float)

    good = np.isfinite(values)
    if not good.any():
        return None  # the nearest cell is land all the way down

    return (
        depths[good],
        values[good],
        float(column[lat_name].values),
        float(column[lon_name].values),
    )


def get_selected_depth(ds: xr.Dataset, depth: float) -> float | None:
    """The actual depth level `get_grid_at_depth` would snap to.

    None for a 2D field: there is no level, and reporting 0 m would be false for
    mixed layer depth, whose values are themselves depths.
    """
    depth_name = _depth_dim(ds)
    if depth_name is None:
        return None
    return float(ds[depth_name].sel({depth_name: depth}, method="nearest").values)


def cache_stats() -> dict:
    """Hit/miss counters and what is currently held, for debugging and tests."""
    with _cache_lock:
        stats = {
            "entries": len(_cache),
            "max_entries": CACHE_MAX_ENTRIES,
            "hits": _cache_hits,
            "misses": _cache_misses,
            "cached_files": [", ".join(p.name for p in k[0]) for k in _cache],
        }
    with _grid_cache_lock:
        stats["grid"] = {
            "entries": len(_grid_cache),
            "max_entries": GRID_CACHE_MAX_ENTRIES,
            "hits": _grid_hits,
            "misses": _grid_misses,
        }
    return stats


def clear_cache() -> None:
    """Drop every cached dataset and grid, and reset the counters."""
    global _cache_hits, _cache_misses, _grid_hits, _grid_misses
    with _cache_lock:
        for dataset in _cache.values():
            dataset.close()
        _cache.clear()
        _cache_hits = 0
        _cache_misses = 0
    with _grid_cache_lock:
        _grid_cache.clear()
        _grid_hits = 0
        _grid_misses = 0
