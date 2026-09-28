"""
Dataset preparation for the statistical ocean-anomaly module.

This layer does no machine learning. It turns the Copernicus files already on
disk into an aligned, real-values-only feature matrix that a later phase can
fit an Isolation Forest to:

    X = [temperature, salinity, current_speed]     (thetao, so, sqrt(uo^2+vo^2))

Three decisions shape everything below.

**All NetCDF reading goes through app/services/ocean_model.py.** That module
owns filename resolution, the Arabian Sea / Bay of Bengal merge and the removal
of the 78E column both regions share. Opening the regional files here would
double-count that column and would drift from the map endpoints the moment the
naming scheme changes.

**Dates are exact, never nearest.** ocean_model.available_days() derives days
from *filenames*, and ocean_model._select_day() resolves them with
method="nearest" -- deliberately, so the map can fall back to the closest
downloaded day instead of rendering blank. That is the wrong behaviour here: a
file whose time axis has a hole would let the trainer use one timestep twice
while believing it had two distinct days, quietly destroying the leave-one-out
guarantee. Every day used here is checked against the file's real time
coordinate first, and rejected -- with a reason -- if it is not genuinely there.

**Slices are cached, not datasets.** ocean_model's dataset cache holds two
entries of ~15.8 MB each. One prepare_anomaly_dataset() call touches four
variables across up to seven days, which would evict the grid and currents
endpoints' working set on every request. That cache is left alone; this module
keeps its own cache of single-depth 2D slices, roughly 0.4 MB each.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from app.services import ocean_model

# Package-internal reuse: _coord_name is how every other reader in
# ocean_model.py resolves latitude/longitude/depth naming. Duplicating it here
# would mean two places to fix if a future product spells a coordinate
# differently.
from app.services.ocean_model import DatasetNotFoundError, _coord_name
from app.services.netcdf_lock import NETCDF_LOCK

# The four CF variables the anomaly module reads. Order matters only for
# readable error messages; thetao is used as the grid reference.
ML_VARIABLES: tuple[str, ...] = ("thetao", "so", "uo", "vo")

# Columns of X, in order. Latitude, longitude, date and depth are deliberately
# NOT features in v1 -- a model given lat/lon learns geography, and one given
# depth learns the thermocline, neither of which is what "unusual for this
# depth, on this day" means. They ride along as metadata on the scoring rows.
FEATURE_NAMES: tuple[str, ...] = ("temperature", "salinity", "current_speed")

# Below this many *other* valid dates, holding the scoring date out would leave
# a baseline too thin to mean anything, so it is folded back in and the caller
# is told. Two is the minimum at which "other days" is plural.
MIN_TRAINING_DATES_FOR_HOLDOUT = 2

TRAINING_STRATEGY_HOLDOUT = "leave_one_date_out"
TRAINING_STRATEGY_FALLBACK = "included_scoring_date_due_to_insufficient_history"

# One full-domain slice is 241 x 409 float32 = 394 KB, plus its coordinate
# vectors -- call it 400 KB. Two things set the cap.
#
# The floor: a single prepare_anomaly_dataset() call must never evict its own
# work, so it needs at least len(ML_VARIABLES) * (valid days) entries, which is
# 4 * 7 = 28 today and grows by four per day downloaded.
#
# The rest is working set. A depth slider is dragged across a handful of levels
# and dragged back, and each level costs another 28 entries; at 48 a two-depth
# comparison already evicted the first depth before returning to it. 128 holds
# roughly four depths at today's coverage for ~51 MB -- the same order as the
# two 15.8 MB datasets ocean_model already keeps -- and degrades by LRU rather
# than growing, which is the point of having a cap at all.
SLICE_CACHE_MAX_ENTRIES = 128

_slice_cache: "OrderedDict[tuple, DepthSlice]" = OrderedDict()
_slice_cache_lock = threading.Lock()
_slice_hits = 0
_slice_misses = 0

# Time axis and depth levels per file, so exact-date checks and depth snapping
# never re-open a NetCDF. Keyed on (path, mtime, size) so a re-download
# invalidates the entry instead of serving a stale answer.
_file_meta_cache: dict[tuple, tuple[frozenset, np.ndarray]] = {}
_file_meta_lock = threading.Lock()


class ExactDateUnavailableError(LookupError):
    """The requested calendar day is not genuinely on the files' time axis.

    Distinct from DatasetNotFoundError, which means no file claims the day at
    all. This one fires when a filename claims a day the data does not contain,
    which is exactly the silent-substitution case this module refuses.
    """


class GridMismatchError(RuntimeError):
    """Two variables' depth slices do not describe the same grid."""


class InsufficientDataError(RuntimeError):
    """Not enough real data to build a training or scoring set."""


# --------------------------------------------------------------------------
# Exact-date verification
# --------------------------------------------------------------------------


def _as_date(value) -> date:
    """Normalise a numpy/cftime/pandas timestamp to a plain calendar date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _parse_day(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"invalid date {value!r} -- expected YYYY-MM-DD")


def _file_metadata(path: Path) -> tuple[frozenset, np.ndarray]:
    """(calendar days actually on the time axis, depth levels) for one file.

    Reads coordinates only. xarray opens lazily, so this never pulls field data
    off disk -- it is cheap enough to call for every candidate day.
    """
    stat = path.stat()
    signature = (str(path), stat.st_mtime_ns, stat.st_size)

    with _file_meta_lock:
        hit = _file_meta_cache.get(signature)
    if hit is not None:
        return hit

    with NETCDF_LOCK, xr.open_dataset(path) as handle:
        if "time" in handle.coords or "time" in handle.dims:
            days = frozenset(
                _as_date(stamp) for stamp in np.atleast_1d(handle["time"].values)
            )
        else:
            days = frozenset()
        depth_name = _coord_name(handle, "depth", "elevation")
        levels = np.array(handle[depth_name].values, dtype=float)

    with _file_meta_lock:
        _file_meta_cache[signature] = (days, levels)
    return days, levels


def verify_exact_date(
    variable: str, day: str | date, data_dir: Path | None = None
) -> tuple[bool, str]:
    """Is this calendar day genuinely present for this variable?

    Returns (ok, reason). Every regional file resolved for the day must contain
    it -- if the Arabian Sea file has 2026-09-25 but the Bay of Bengal file does
    not, the merged grid would be half real and half substituted, so the day is
    rejected rather than half-used.
    """
    wanted = _parse_day(day)
    nc_var = ocean_model.resolve_nc_variable(variable)

    try:
        paths = ocean_model.find_dataset_paths(nc_var, wanted, data_dir)
    except DatasetNotFoundError:
        return False, f"{nc_var} {wanted}: no downloaded file covers this day"

    missing = [path.name for path in paths if wanted not in _file_metadata(path)[0]]
    if missing:
        return False, (
            f"{nc_var} {wanted}: the filename claims this day but it is absent "
            f"from the time axis of {missing}"
        )
    return True, ""


def valid_dates(
    variable: str, data_dir: Path | None = None
) -> tuple[list[date], list[dict]]:
    """Days this variable can serve exactly, and the ones dropped with reasons.

    available_days() is the candidate list -- it is filename-derived, so it is a
    claim, not a fact. Each claim is then checked against the real time axis.
    """
    kept: list[date] = []
    dropped: list[dict] = []
    for day in ocean_model.available_days(variable, data_dir):
        ok, reason = verify_exact_date(variable, day, data_dir)
        if ok:
            kept.append(day)
        else:
            dropped.append(
                {"date": str(day), "variable": variable, "reason": reason}
            )
    return kept, dropped


def common_valid_dates(
    variables: Sequence[str] = ML_VARIABLES, data_dir: Path | None = None
) -> tuple[list[date], list[dict]]:
    """Days every variable can serve exactly, plus everything excluded and why.

    A row needs all four variables at the same cell on the same day, so the
    usable set is the intersection. Days that survive verification for some
    variables but not others are reported too -- otherwise a partially
    downloaded day would just quietly vanish from the training set.
    """
    per_variable: dict[str, set] = {}
    dropped: list[dict] = []

    for variable in variables:
        kept, drops = valid_dates(variable, data_dir)
        per_variable[variable] = set(kept)
        dropped.extend(drops)

    if not per_variable:
        return [], dropped

    common = set.intersection(*per_variable.values())
    union: set = set().union(*per_variable.values())

    for day in sorted(union - common):
        absent = sorted(v for v in variables if day not in per_variable[v])
        dropped.append(
            {
                "date": str(day),
                "variable": ",".join(absent),
                "reason": (
                    f"not exactly available for every variable "
                    f"{list(variables)}; missing from {absent}"
                ),
            }
        )

    return sorted(common), dropped


# --------------------------------------------------------------------------
# Depth slices
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DepthSlice:
    """One variable, one exact day, one model level, as a 2D (lat, lon) field.

    `values` is a copy rather than an xarray view on purpose: a view into the
    (40, lat, lon) cube would pin the whole ~15.8 MB dataset in this cache
    instead of the ~0.4 MB slice. The arrays are marked read-only because the
    instance is shared between callers straight out of the cache.
    """

    variable: str
    date: str
    requested_depth: float
    actual_depth: float
    latitudes: np.ndarray
    longitudes: np.ndarray
    values: np.ndarray

    @property
    def nbytes(self) -> int:
        return self.values.nbytes + self.latitudes.nbytes + self.longitudes.nbytes


def _depth_levels_for(
    variable: str, day: date, data_dir: Path | None = None
) -> np.ndarray:
    """The vertical axis for a variable on a day, from file metadata only.

    Regions are downloaded separately, so their axes are compared rather than
    assumed equal -- snapping a depth against one region and reading it from
    another would silently mix levels.
    """
    paths = ocean_model.find_dataset_paths(variable, day, data_dir)
    levels = _file_metadata(paths[0])[1]
    for path in paths[1:]:
        other = _file_metadata(path)[1]
        if not np.array_equal(levels, other):
            raise GridMismatchError(
                f"{variable} on {day}: {paths[0].name} and {path.name} have "
                f"different depth axes ({levels.size} vs {other.size} levels)"
            )
    return levels


def snap_depth(levels: np.ndarray, requested_depth: float) -> float:
    """Nearest real model level to a requested depth.

    Depth *may* snap -- the product has 40 discrete, unevenly spaced levels, so
    100 m genuinely means 92.326 m and there is nothing in between to have. This
    is the one axis where "nearest" is honest; date is not.
    """
    if levels.size == 0:
        raise GridMismatchError("dataset has no depth levels")
    return float(levels[int(np.argmin(np.abs(levels - float(requested_depth))))])


def load_depth_slice(
    variable: str,
    date_str: str | date,
    requested_depth: float,
    data_dir: Path | None = None,
) -> DepthSlice:
    """One variable at one exact day and the nearest model level.

    Raises ExactDateUnavailableError if the day is not genuinely on the time
    axis. Repeat calls are served from this module's slice cache, so the four
    variables of a training day cost four dataset reads once and nothing after.
    """
    global _slice_hits, _slice_misses

    nc_var = ocean_model.resolve_nc_variable(variable)
    day = _parse_day(date_str)

    ok, reason = verify_exact_date(nc_var, day, data_dir)
    if not ok:
        raise ExactDateUnavailableError(reason)

    paths = tuple(ocean_model.find_dataset_paths(nc_var, day, data_dir))
    actual_depth = snap_depth(_depth_levels_for(nc_var, day, data_dir), requested_depth)

    # Keyed on the resolved level, not the requested one, so 100 m and 95 m
    # share the single entry they both resolve to. Keyed on the file paths so a
    # re-download invalidates the entry naturally.
    key = (paths, nc_var, str(day), actual_depth)

    with _slice_cache_lock:
        if key in _slice_cache:
            _slice_cache.move_to_end(key)
            _slice_hits += 1
            return _slice_cache[key]

    dataset = ocean_model.load_dataset(nc_var, str(day), data_dir)

    # Belt and braces. verify_exact_date() already cleared this day, but
    # load_dataset() resolves time with method="nearest", so confirm what came
    # back is the day asked for before any of it becomes a training row.
    stamps = {_as_date(stamp) for stamp in np.atleast_1d(dataset["time"].values)}
    if stamps != {day}:
        raise ExactDateUnavailableError(
            f"{nc_var}: asked for {day} but the loaded dataset carries "
            f"{sorted(str(s) for s in stamps)}"
        )

    da = dataset[nc_var]
    if "time" in da.dims:
        da = da.isel(time=0)

    depth_name = _coord_name(da, "depth", "elevation")
    # actual_depth is already one of the levels, so this selects it exactly;
    # method="nearest" only guards float round-tripping.
    da = da.sel({depth_name: actual_depth}, method="nearest")

    lat_name = _coord_name(da, "latitude", "lat")
    lon_name = _coord_name(da, "longitude", "lon")
    da = da.transpose(lat_name, lon_name)

    latitudes = np.array(da[lat_name].values, dtype=np.float64)
    longitudes = np.array(da[lon_name].values, dtype=np.float64)
    values = np.array(da.values, dtype=np.float32)  # copies, see DepthSlice
    for array in (latitudes, longitudes, values):
        array.setflags(write=False)

    depth_slice = DepthSlice(
        variable=nc_var,
        date=str(day),
        requested_depth=float(requested_depth),
        actual_depth=float(da[depth_name].values),
        latitudes=latitudes,
        longitudes=longitudes,
        values=values,
    )

    with _slice_cache_lock:
        if key in _slice_cache:
            # Another thread won the race; keep its copy so callers share one.
            _slice_cache.move_to_end(key)
            _slice_hits += 1
            return _slice_cache[key]
        _slice_cache[key] = depth_slice
        _slice_misses += 1
        while len(_slice_cache) > SLICE_CACHE_MAX_ENTRIES:
            _slice_cache.popitem(last=False)  # drop least-recently-used

    return depth_slice


def slice_cache_stats() -> dict:
    """Hit/miss counters and current footprint, for the test script and /debug."""
    with _slice_cache_lock:
        held = list(_slice_cache.values())
        return {
            "entries": len(held),
            "max_entries": SLICE_CACHE_MAX_ENTRIES,
            "hits": _slice_hits,
            "misses": _slice_misses,
            "bytes": sum(s.nbytes for s in held),
        }


def clear_cache() -> None:
    """Drop every cached slice and file-metadata entry, and reset counters."""
    global _slice_hits, _slice_misses
    with _slice_cache_lock:
        _slice_cache.clear()
        _slice_hits = 0
        _slice_misses = 0
    with _file_meta_lock:
        _file_meta_cache.clear()


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------


@dataclass(eq=False)
class AnomalyDataset:
    """Everything a later fitting phase needs, and nothing it should not have.

    X_train and X_score share the FEATURE_NAMES column order. Latitude and
    longitude are carried only for the scoring rows, so a caller can put a score
    back on the map -- they are not, and must not become, model inputs.
    """

    # training
    X_train: np.ndarray
    # scoring
    X_score: np.ndarray
    score_latitudes: np.ndarray
    score_longitudes: np.ndarray
    # metadata
    requested_depth: float
    actual_depth: float
    scoring_date: str
    training_dates: list[str]
    excluded_scoring_date: bool
    training_strategy: str
    training_sample_count: int
    scoring_sample_count: int
    training_rows_per_date: dict[str, int]
    dropped_dates: list[dict]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def metadata(self) -> dict:
        """The non-array half, ready to become an API response later."""
        return {
            "requested_depth": self.requested_depth,
            "actual_depth": self.actual_depth,
            "scoring_date": self.scoring_date,
            "training_dates": list(self.training_dates),
            "excluded_scoring_date": self.excluded_scoring_date,
            "training_strategy": self.training_strategy,
            "training_sample_count": self.training_sample_count,
            "scoring_sample_count": self.scoring_sample_count,
            "training_rows_per_date": dict(self.training_rows_per_date),
            "dropped_dates": list(self.dropped_dates),
            "feature_names": list(self.feature_names),
        }


def build_feature_rows(
    day: str | date, requested_depth: float, data_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One day's rows: (features (n,3), lats (n,), lons (n,), actual_depth).

    Every row is a cell where all four variables hold a real number. The audit
    found thetao/so/uo/vo sharing an identical wet mask at every level, but that
    is a property of today's files rather than a guarantee, and a half-known row
    is not a row -- so the mask is intersected explicitly.
    """
    slices = {
        variable: load_depth_slice(variable, day, requested_depth, data_dir)
        for variable in ML_VARIABLES
    }
    reference = slices[ML_VARIABLES[0]]

    for variable, depth_slice in slices.items():
        if not np.array_equal(depth_slice.latitudes, reference.latitudes):
            raise GridMismatchError(
                f"{variable} on {day}: latitude axis differs from "
                f"{reference.variable} ({depth_slice.latitudes.size} vs "
                f"{reference.latitudes.size} points)"
            )
        if not np.array_equal(depth_slice.longitudes, reference.longitudes):
            raise GridMismatchError(
                f"{variable} on {day}: longitude axis differs from "
                f"{reference.variable} ({depth_slice.longitudes.size} vs "
                f"{reference.longitudes.size} points)"
            )
        if depth_slice.actual_depth != reference.actual_depth:
            raise GridMismatchError(
                f"{variable} on {day}: resolved to {depth_slice.actual_depth} m "
                f"but {reference.variable} resolved to {reference.actual_depth} m"
            )

    temperature = slices["thetao"].values
    salinity = slices["so"].values
    eastward = slices["uo"].values
    northward = slices["vo"].values

    mask = (
        np.isfinite(temperature)
        & np.isfinite(salinity)
        & np.isfinite(eastward)
        & np.isfinite(northward)
    )

    # Masked first, then squared: keeps NaN out of the arithmetic entirely
    # rather than computing over land and discarding the result.
    u = eastward[mask]
    v = northward[mask]
    current_speed = np.sqrt(u * u + v * v)

    features = np.column_stack(
        [temperature[mask], salinity[mask], current_speed]
    ).astype(np.float32, copy=False)

    lat_grid, lon_grid = np.meshgrid(
        reference.latitudes, reference.longitudes, indexing="ij"
    )
    return features, lat_grid[mask], lon_grid[mask], reference.actual_depth


def prepare_anomaly_dataset(
    scoring_date: str | date,
    requested_depth: float,
    *,
    data_dir: Path | None = None,
) -> AnomalyDataset:
    """Training and scoring matrices for one depth and one day.

    The scoring day is held out of training whenever at least
    MIN_TRAINING_DATES_FOR_HOLDOUT other exactly-available days exist, so a cell
    is judged against days it did not itself contribute to. Below that, the
    baseline would be too thin to hold out from, so the scoring day is folded
    back in and training_strategy says so.

    Nothing is fitted here and nothing is scaled -- this returns real Copernicus
    values in their native units.
    """
    scoring_day = _parse_day(scoring_date)
    valid, dropped = common_valid_dates(ML_VARIABLES, data_dir)

    if not valid:
        raise InsufficientDataError(
            f"no day is exactly available for all of {list(ML_VARIABLES)}; "
            f"dropped: {dropped or 'nothing'}"
        )

    if scoring_day not in valid:
        raise ExactDateUnavailableError(
            f"{scoring_day} is not exactly available for every variable "
            f"{list(ML_VARIABLES)}. Exactly available: "
            f"{[str(day) for day in valid]}"
        )

    others = [day for day in valid if day != scoring_day]
    if len(others) >= MIN_TRAINING_DATES_FOR_HOLDOUT:
        training_days = others
        excluded_scoring_date = True
        training_strategy = TRAINING_STRATEGY_HOLDOUT
    else:
        training_days = list(valid)
        excluded_scoring_date = False
        training_strategy = TRAINING_STRATEGY_FALLBACK

    blocks: list[np.ndarray] = []
    rows_per_date: dict[str, int] = {}
    actual_depth: float | None = None

    for day in training_days:
        features, _, _, depth_used = build_feature_rows(day, requested_depth, data_dir)
        if actual_depth is None:
            actual_depth = depth_used
        elif depth_used != actual_depth:
            raise GridMismatchError(
                f"{day} resolved {requested_depth} m to {depth_used} m while "
                f"earlier days resolved it to {actual_depth} m"
            )
        blocks.append(features)
        rows_per_date[str(day)] = int(features.shape[0])

    X_train = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )

    X_score, score_lats, score_lons, score_depth = build_feature_rows(
        scoring_day, requested_depth, data_dir
    )
    if actual_depth is None:
        actual_depth = score_depth
    elif score_depth != actual_depth:
        raise GridMismatchError(
            f"scoring day {scoring_day} resolved {requested_depth} m to "
            f"{score_depth} m while training days resolved it to {actual_depth} m"
        )

    if X_train.shape[0] == 0:
        raise InsufficientDataError(
            f"no training rows at {actual_depth} m -- every cell is land or "
            f"missing on {[str(day) for day in training_days]}"
        )
    if X_score.shape[0] == 0:
        raise InsufficientDataError(
            f"no scoring rows at {actual_depth} m on {scoring_day} -- every cell "
            f"is land or missing"
        )

    return AnomalyDataset(
        X_train=X_train,
        X_score=X_score,
        score_latitudes=score_lats,
        score_longitudes=score_lons,
        requested_depth=float(requested_depth),
        actual_depth=float(actual_depth),
        scoring_date=str(scoring_day),
        training_dates=[str(day) for day in training_days],
        excluded_scoring_date=excluded_scoring_date,
        training_strategy=training_strategy,
        training_sample_count=int(X_train.shape[0]),
        scoring_sample_count=int(X_score.shape[0]),
        training_rows_per_date=rows_per_date,
        dropped_dates=dropped,
        feature_names=FEATURE_NAMES,
    )


# ==========================================================================
# Isolation Forest
# ==========================================================================
#
# What this does and does not claim
# ---------------------------------
# The forest learns what a (temperature, salinity, current_speed) triple
# normally looks like at ONE model level, across the handful of other days
# currently downloaded, and flags the cells on the scored day that are hardest
# to explain from that baseline. That is:
#
#     statistical anomaly detection relative to the loaded short-term model
#     baseline
#
# It is NOT climatological or seasonal anomaly detection, and it is not hazard,
# cyclone or disaster prediction. With six other days on disk the baseline is a
# single week, so a feature that persisted all week -- a standing eddy, a
# monsoon freshwater plume -- is *normal* to this model by construction.
#
# The row counts are large (~392k at 92 m) but they are neighbouring cells of
# one numerical grid, not independent physical observations. Six days of one
# model is the real sample size; treating 392k as a degrees-of-freedom argument
# would be wrong.

from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

MODEL_NAME = "IsolationForest"

DEFAULT_N_ESTIMATORS = 200
DEFAULT_CONTAMINATION = 0.02
# A cap, not a promise: the effective value is min(this, len(X_train)).
DEFAULT_MAX_SAMPLES = 2048
# Fixed so the same request returns the same map. A colour that changes on
# refresh reads as a bug, whatever the caveats say.
RANDOM_STATE = 42

# Below 0.1% the flagged set is too small to be worth drawing; above 10% the
# word "outlier" stops meaning anything.
MIN_CONTAMINATION = 0.001
MAX_CONTAMINATION = 0.10

# Presentation bands for anomaly_percentile. These rank a point against the
# rest of the field on the day being viewed. They are UI ordering labels, not
# scientific hazard classes, and nothing downstream should treat them as
# thresholds for action.
SEVERITY_BANDS: tuple[tuple[float, str], ...] = (
    (99.0, "very_high"),
    (98.0, "high"),
    (95.0, "moderate"),
)
BASELINE_SEVERITY = "normal"

# What feature_context is, in one place, because it is the number most likely
# to be over-read. Carried in the API response so the wording travels with it.
FEATURE_CONTEXT_NOTE = (
    "training_percentile is the share of the training baseline at or below "
    "this cell's value for that feature; z_score is the same value expressed "
    "in standard deviations of the training baseline, via the scaler the model "
    "was fitted with. Both are descriptive statistics of one feature against "
    "the baseline. They are not feature importance, not a contribution, not a "
    "causal explanation and not SHAP values -- Isolation Forest does not "
    "report which feature caused a cell to be isolated."
)


INTERPRETATION = {
    "what_it_is": (
        "statistical anomaly detection relative to the loaded short-term model "
        "baseline"
    ),
    "what_it_is_not": [
        "climatological anomaly detection",
        "seasonal anomaly detection",
        "hazard prediction",
        "cyclone detection",
        "disaster prediction",
    ],
    "sample_note": (
        "training rows are spatially correlated cells of one numerical model "
        "grid, not independent physical observations"
    ),
    "feature_context_note": FEATURE_CONTEXT_NOTE,
    "severity_note": (
        "severity and anomaly_percentile rank a cell against the rest of the "
        "scored field on this day; they are presentation labels, not "
        "probabilities, confidences or hazard levels"
    ),
}

# Measured, not estimated: at the defaults (200 trees, max_samples=2048) the
# trees hold 289-1267 nodes each and the fitted pair occupies ~11 MB in memory
# (~13.5 MB pickled). The scaler is negligible beside that -- three means and
# three standard deviations.
#
# Four entries caps this near 44 MB, which covers the depths a user moves
# between in one sitting. Going wider is tempting and wrong: the cost scales
# with n_estimators x max_samples, so raising either hyper-parameter inflates
# every entry at once, and this cache sits alongside the slice cache (~51 MB)
# and ocean_model's datasets (~32 MB) in the same process.
MODEL_CACHE_MAX_ENTRIES = 4

_model_cache: "OrderedDict[tuple, FittedAnomalyModel]" = OrderedDict()
_model_cache_lock = threading.Lock()
_model_hits = 0
_model_misses = 0

# The meridian the two basin downloads share, used to name a point's basin.
BASIN_SPLIT_LONGITUDE = 78.0

# --------------------------------------------------------------------------
# Validated prototype depth range
# --------------------------------------------------------------------------
#
# This module will happily prepare and score ANY model level -- that capability
# is needed to keep auditing the ones that are not trusted yet. The range below
# is a separate, narrower claim: the depths whose output has actually been
# checked for geographic artefacts and found defensible. Callers that publish
# results (the API layer) enforce it; internal tests deliberately do not.
#
# The Phase 3 audit measured, per depth, how over-represented each subset of
# cells is among the flagged ones (enrichment = share of outliers / share of
# cells; 1.0 is neutral):
#
#   0.494 m  low-salinity cells 15.1x, coastal 9.3x, basin boundary 2.5x.
#            Every one of the top 20 sat at 21.6-22.0N, 87.2-88.7E with
#            salinity 4.5-6.9 PSU -- the Ganges-Brahmaputra-Meghna plume. The
#            model was rediscovering a river mouth, not finding an anomaly.
#
#   92.326 m no subset over-represented by 2x. Bay of Bengal 1.84x, basin
#            boundary 0.00x, domain edge 0.35x, coastal 0.35x, low salinity
#            1.00x. The flagged cells were a coherent subsurface jet.
#
#   541.089 m  domain edge 2.5x, and all top 20 sat on the western wall of the
#              Arabian Sea download box at 62.0-62.6E.
#   1062.440 m domain edge 3.1x, low salinity 5.7x.
#
# So the deep artefacts track the edge of the DOWNLOAD, not the ocean: pooling
# both basins over seven days makes a boundary water mass look rare. Until that
# is addressed, publishing those depths would be presenting an artefact of the
# crop as a finding.
VALIDATED_DEPTH_MIN_M = 50.0
VALIDATED_DEPTH_MAX_M = 250.0

# Discrete levels mean the snap can land slightly outside the band: a 50 m
# request resolves to 47.374 m and a 250 m request to 266.040 m, because those
# are the nearest levels that exist. The guard is on what was asked for.
VALIDATED_DEPTH_RANGE_NOTE = (
    "Validated for requested depths between "
    f"{VALIDATED_DEPTH_MIN_M:g} m and {VALIDATED_DEPTH_MAX_M:g} m. Outside that "
    "band the seven-day pooled-domain baseline flags artefacts of the download "
    "crop rather than ocean states: at the surface the Ganges-Brahmaputra-Meghna "
    "freshwater plume dominates (low-salinity cells over-represented 15x), and "
    "below ~500 m the flagged cells concentrate on the western edge of the "
    "Arabian Sea download box (over-represented 2.5-3.1x)."
)


def depth_is_validated(requested_depth: float) -> bool:
    """Has this requested depth been audited for geographic artefacts?

    Judged on the depth ASKED FOR, not the level it snaps to -- the levels are
    discrete, so 50 m resolves to 47.374 m and 250 m to 266.040 m, and rejecting
    a request for landing a few metres outside its own band would be arbitrary.
    """
    return VALIDATED_DEPTH_MIN_M <= float(requested_depth) <= VALIDATED_DEPTH_MAX_M


class InvalidModelConfigError(ValueError):
    """A model hyper-parameter is outside the range this service accepts."""


@dataclass(frozen=True)
class AnomalyModelConfig:
    """Hyper-parameters, kept in one hashable place so they can key the cache."""

    n_estimators: int = DEFAULT_N_ESTIMATORS
    contamination: float = DEFAULT_CONTAMINATION
    max_samples: int = DEFAULT_MAX_SAMPLES
    random_state: int = RANDOM_STATE

    def validated(self) -> "AnomalyModelConfig":
        if not MIN_CONTAMINATION <= self.contamination <= MAX_CONTAMINATION:
            raise InvalidModelConfigError(
                f"contamination must be between {MIN_CONTAMINATION} and "
                f"{MAX_CONTAMINATION}, got {self.contamination}"
            )
        if self.n_estimators < 1:
            raise InvalidModelConfigError(
                f"n_estimators must be at least 1, got {self.n_estimators}"
            )
        if self.max_samples < 1:
            raise InvalidModelConfigError(
                f"max_samples must be at least 1, got {self.max_samples}"
            )
        return self

    def effective_max_samples(self, n_train: int) -> int:
        """The cap, clipped to the data actually available."""
        return int(min(self.max_samples, n_train))


@dataclass(eq=False)
class FittedAnomalyModel:
    """A scaler and forest fitted together, plus what they were fitted on."""

    scaler: StandardScaler
    forest: IsolationForest
    config: AnomalyModelConfig
    effective_max_samples: int
    actual_depth: float
    training_dates: list[str]
    training_sample_count: int
    fit_seconds: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _training_fingerprint(
    training_days: Sequence[str], data_dir: Path | None = None
) -> tuple:
    """Identity of every file behind the training set.

    Path names alone are not enough: re-running the downloader writes the same
    filename with new content, and a model fitted on the old bytes must not be
    served for the new ones. Size and mtime make that visible.
    """
    signatures = set()
    for variable in ML_VARIABLES:
        for day in training_days:
            for path in ocean_model.find_dataset_paths(variable, day, data_dir):
                stat = path.stat()
                signatures.add((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(signatures))


def _model_cache_key(
    dataset: AnomalyDataset,
    config: AnomalyModelConfig,
    effective_max_samples: int,
    data_dir: Path | None,
) -> tuple:
    return (
        round(dataset.actual_depth, 6),
        tuple(dataset.training_dates),
        _training_fingerprint(dataset.training_dates, data_dir),
        tuple(dataset.feature_names),
        config.n_estimators,
        float(config.contamination),
        effective_max_samples,
        config.random_state,
    )


def fit_anomaly_model(
    dataset: AnomalyDataset,
    config: AnomalyModelConfig | None = None,
    *,
    data_dir: Path | None = None,
) -> FittedAnomalyModel:
    """Fit the scaler and forest on the training rows, or return a cached pair.

    **The scaler is fitted on X_train only** -- the scoring day never
    contributes a mean or a standard deviation, or its own values would be
    folded into the baseline it is being judged against.

    Standardisation is here for pipeline consistency, so that feature context
    can later be reported in comparable units, and so a future model that *is*
    scale-sensitive can be swapped in without changing the surrounding code. It
    is not load-bearing for this estimator: Isolation Forest splits on axis
    values rather than distances, and an affine rescaling of one column leaves
    the tree structure essentially unchanged. Scaling does not recover
    information from a low-variance feature -- if salinity barely varies at
    541 m, dividing by its standard deviation does not make it more informative,
    it only re-expresses it.

    X_train is not subsampled by hand. IsolationForest already draws
    max_samples rows per tree, which is the sampling this algorithm is designed
    around; a second layer on top would just discard data for no benefit.
    """
    global _model_hits, _model_misses

    config = (config or AnomalyModelConfig()).validated()
    effective_max_samples = config.effective_max_samples(dataset.training_sample_count)

    # A holdout dataset must never have handed us the scoring day's rows.
    if (
        dataset.training_strategy == TRAINING_STRATEGY_HOLDOUT
        and dataset.scoring_date in dataset.training_dates
    ):
        raise InsufficientDataError(
            f"refusing to fit: strategy is {TRAINING_STRATEGY_HOLDOUT} but the "
            f"scoring date {dataset.scoring_date} is in the training dates"
        )

    key = _model_cache_key(dataset, config, effective_max_samples, data_dir)

    with _model_cache_lock:
        if key in _model_cache:
            _model_cache.move_to_end(key)
            _model_hits += 1
            return _model_cache[key]

    started = time.perf_counter()

    scaler = StandardScaler()
    scaler.fit(dataset.X_train)
    X_train_scaled = scaler.transform(dataset.X_train)

    forest = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        max_samples=effective_max_samples,
        random_state=config.random_state,
        n_jobs=-1,
    )
    forest.fit(X_train_scaled)

    fitted = FittedAnomalyModel(
        scaler=scaler,
        forest=forest,
        config=config,
        effective_max_samples=effective_max_samples,
        actual_depth=dataset.actual_depth,
        training_dates=list(dataset.training_dates),
        training_sample_count=dataset.training_sample_count,
        fit_seconds=time.perf_counter() - started,
    )

    with _model_cache_lock:
        if key in _model_cache:
            _model_cache.move_to_end(key)
            _model_hits += 1
            return _model_cache[key]
        _model_cache[key] = fitted
        _model_misses += 1
        while len(_model_cache) > MODEL_CACHE_MAX_ENTRIES:
            _model_cache.popitem(last=False)  # drop least-recently-used

    return fitted


def model_cache_stats() -> dict:
    """Entry count and hit/miss counters for the fitted-model cache."""
    with _model_cache_lock:
        return {
            "entries": len(_model_cache),
            "max_entries": MODEL_CACHE_MAX_ENTRIES,
            "hits": _model_hits,
            "misses": _model_misses,
        }


def clear_model_cache() -> None:
    """Drop every fitted model and reset the counters."""
    global _model_hits, _model_misses
    with _model_cache_lock:
        _model_cache.clear()
        _model_hits = 0
        _model_misses = 0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def anomaly_percentiles(raw_scores: np.ndarray) -> np.ndarray:
    """Rank each cell against the rest of the field it was scored with.

    Definition, exactly: for cell i,

        anomaly_percentile(i) = 100 * |{j : raw_score(j) >= raw_score(i)}| / n

    where n is the number of scored cells and raw_score is sklearn's
    score_samples. sklearn's convention is that LOWER score_samples means more
    isolated, so the most anomalous cell has the smallest raw score, every other
    cell scores at least as high as it, and it lands at exactly 100.0. The least
    anomalous cell lands at 100/n. Ties share a value.

    This is a rank within one day's field and nothing more. It is not a
    probability that anything is wrong, not a confidence, and not a hazard
    level: on a completely unremarkable day some cell is still the most unusual
    one present, and it still scores 100.
    """
    n = int(raw_scores.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(raw_scores)
    # Number of cells scoring strictly lower (i.e. strictly more anomalous).
    strictly_lower = np.searchsorted(ordered, raw_scores, side="left")
    return 100.0 * (n - strictly_lower) / n


def severity_for(percentile: float) -> str:
    """Presentation band for one percentile. UI ordering only."""
    for threshold, label in SEVERITY_BANDS:
        if percentile >= threshold:
            return label
    return BASELINE_SEVERITY


@dataclass(eq=False)
class ScoredField:
    """Every scored cell, before the outliers are separated out."""

    raw_scores: np.ndarray
    decision_scores: np.ndarray
    predictions: np.ndarray
    percentiles: np.ndarray
    is_outlier: np.ndarray
    score_seconds: float


def score_field(fitted: FittedAnomalyModel, dataset: AnomalyDataset) -> ScoredField:
    """Run the three sklearn score functions over the scoring rows.

    * score_samples  -- raw isolation score; LOWER is more anomalous.
    * decision_function -- the same quantity shifted by the fitted offset, so
      negative marks an outlier at the contamination the forest was fitted with.
    * predict -- -1 for an outlier, +1 otherwise; equivalent to
      decision_function < 0.
    """
    started = time.perf_counter()
    X_score_scaled = fitted.transform(dataset.X_score)

    raw_scores = fitted.forest.score_samples(X_score_scaled)
    decision_scores = fitted.forest.decision_function(X_score_scaled)
    predictions = fitted.forest.predict(X_score_scaled)

    return ScoredField(
        raw_scores=raw_scores,
        decision_scores=decision_scores,
        predictions=predictions,
        percentiles=anomaly_percentiles(raw_scores),
        is_outlier=(predictions == -1),
        score_seconds=time.perf_counter() - started,
    )


@dataclass(eq=False)
class AnomalyPoint:
    """One flagged cell, with the real values that got it flagged.

    `feature_context` places each of the three values against the SAME X_train
    the forest was fitted on. See feature_context_for() for exactly what the
    two numbers mean, and what they are not.
    """

    lat: float
    lon: float
    temperature: float
    salinity: float
    current_speed: float
    raw_score: float
    decision_function: float
    anomaly_percentile: float
    severity: str
    feature_context: dict

    def as_dict(self) -> dict:
        return {
            "lat": round(self.lat, 3),
            "lon": round(self.lon, 3),
            "temperature": round(self.temperature, 3),
            "salinity": round(self.salinity, 3),
            "current_speed": round(self.current_speed, 4),
            "raw_score": round(self.raw_score, 6),
            "decision_function": round(self.decision_function, 6),
            "anomaly_percentile": round(self.anomaly_percentile, 3),
            "severity": self.severity,
            "feature_context": self.feature_context,
        }


def feature_context_for(
    dataset: AnomalyDataset,
    fitted: FittedAnomalyModel,
    rows: np.ndarray,
) -> list[dict]:
    """Place each feature of each flagged row against the training baseline.

    Two numbers per feature, both computed from the exact X_train that fitted
    this model:

      training_percentile -- 100 * |{t in X_train[:, f] : t <= v}| / n_train.
                             The share of the baseline this cell sits at or
                             above for that feature. 99.8 means only 0.2% of
                             baseline cells had a higher value.
      z_score             -- (v - mean) / scale, using the fitted
                             StandardScaler's own statistics, so it is the
                             baseline's spread and not the scored day's.

    Neither says which feature made the forest isolate the cell. A cell can sit
    at the 50th percentile on all three and still be flagged, because isolation
    responds to the COMBINATION; and a feature at the 99.8th percentile may be
    incidental. See FEATURE_CONTEXT_NOTE.
    """
    if rows.size == 0:
        return []

    values = dataset.X_score[rows].astype(np.float64)
    n_train = dataset.X_train.shape[0]

    # Sorting three columns of ~392k costs ~15 ms and is not worth caching
    # against a ~700 ms request; keeping it here also keeps the fitted-model
    # cache free of a 4.7 MB copy per entry.
    ordered = np.sort(dataset.X_train, axis=0)

    contexts: list[dict] = []
    percentiles = np.empty(values.shape, dtype=np.float64)
    for column in range(values.shape[1]):
        at_or_below = np.searchsorted(
            ordered[:, column], values[:, column], side="right"
        )
        percentiles[:, column] = 100.0 * at_or_below / n_train

    z_scores = (values - fitted.scaler.mean_) / fitted.scaler.scale_

    for index in range(values.shape[0]):
        contexts.append(
            {
                name: {
                    "training_percentile": round(float(percentiles[index, column]), 3),
                    "z_score": round(float(z_scores[index, column]), 3),
                }
                for column, name in enumerate(dataset.feature_names)
            }
        )
    return contexts


@dataclass(eq=False)
class AnomalyResult:
    """What a later API layer will serialise: the flagged cells and provenance.

    The ~65k scored cells are deliberately absent. Only the flagged minority is
    carried; score_all_points() exists for debugging when the full field is
    genuinely wanted.
    """

    model_name: str
    feature_names: tuple[str, ...]
    requested_depth: float
    actual_depth: float
    scoring_date: str
    training_dates: list[str]
    training_strategy: str
    excluded_scoring_date: bool
    training_sample_count: int
    scoring_sample_count: int
    n_estimators: int
    max_samples: int
    contamination: float
    random_state: int
    outlier_count: int
    outlier_percentage: float
    outliers: list[AnomalyPoint]
    dropped_dates: list[dict]
    geography: dict
    fit_seconds: float
    score_seconds: float
    interpretation: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.interpretation is None:
            self.interpretation = dict(INTERPRETATION)

    @property
    def metadata(self) -> dict:
        """Everything except the flagged points, ready to become a response."""
        return {
            "model_name": self.model_name,
            "feature_names": list(self.feature_names),
            "requested_depth": self.requested_depth,
            "actual_depth": self.actual_depth,
            "scoring_date": self.scoring_date,
            "training_dates": list(self.training_dates),
            "training_strategy": self.training_strategy,
            "excluded_scoring_date": self.excluded_scoring_date,
            "training_sample_count": self.training_sample_count,
            "scoring_sample_count": self.scoring_sample_count,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "outlier_count": self.outlier_count,
            "outlier_percentage": self.outlier_percentage,
            "dropped_dates": list(self.dropped_dates),
            "interpretation": dict(self.interpretation),
        }


# --------------------------------------------------------------------------
# Geographic bias reporting
# --------------------------------------------------------------------------


def coastal_flags(
    scoring_date: str | date, requested_depth: float, data_dir: Path | None = None
) -> np.ndarray:
    """Per scoring row: is this wet cell adjacent to land?

    Built from the same combined finite mask build_feature_rows() uses and
    flattened in the same C order, so the result lines up row for row. Cells
    outside the grid are treated as water, so this measures proximity to real
    land rather than to the edge of the download -- the domain edge is reported
    separately, from geometry.
    """
    slices = [
        load_depth_slice(variable, scoring_date, requested_depth, data_dir)
        for variable in ML_VARIABLES
    ]
    mask = np.logical_and.reduce([np.isfinite(s.values) for s in slices])

    padded = np.pad(mask, 1, mode="constant", constant_values=True)
    neighbours_wet = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return (mask & ~neighbours_wet)[mask]


def _subset_report(subset: np.ndarray, outliers: np.ndarray) -> dict:
    """How over- or under-represented a subset of cells is among the outliers.

    enrichment is (share of outliers) / (share of cells): 1.0 means the subset
    is flagged at exactly the rate its size would predict, 3.0 means three times
    as often. It is the number that says whether the model has found something
    or has merely rediscovered a region.
    """
    total = int(outliers.size)
    flagged = int(outliers.sum())
    cells = int(subset.sum())
    hits = int((subset & outliers).sum())

    share_of_cells = 100.0 * cells / total if total else 0.0
    share_of_outliers = 100.0 * hits / flagged if flagged else 0.0
    return {
        "cells": cells,
        "outliers": hits,
        "share_of_cells_pct": round(share_of_cells, 3),
        "share_of_outliers_pct": round(share_of_outliers, 3),
        "outlier_rate_pct": round(100.0 * hits / cells, 3) if cells else 0.0,
        "enrichment": (
            round(share_of_outliers / share_of_cells, 3) if share_of_cells else None
        ),
    }


def geographic_summary(
    dataset: AnomalyDataset,
    is_outlier: np.ndarray,
    coastal: np.ndarray | None = None,
) -> dict:
    """Is the model finding unusual states, or just finding the Bay of Bengal?

    Training pools both basins and excludes lat/lon by design, so a state that
    is rare across the whole domain can be perfectly ordinary where it sits --
    Bay of Bengal surface water is genuinely fresher than Arabian Sea water, and
    a model that flags it is describing geography, not anomaly. This reports the
    distribution so the question can be answered from evidence; it does not
    change anything.
    """
    lats = dataset.score_latitudes
    lons = dataset.score_longitudes
    salinity = dataset.X_score[:, 1]

    arabian = lons < BASIN_SPLIT_LONGITUDE
    bengal = ~arabian

    report: dict = {
        "basins": {
            "arabian_sea": _subset_report(arabian, is_outlier),
            "bay_of_bengal": _subset_report(bengal, is_outlier),
        },
        "near_basin_boundary": _subset_report(
            np.abs(lons - BASIN_SPLIT_LONGITUDE) <= 0.5, is_outlier
        ),
        "domain_edge": _subset_report(
            (lats <= lats.min() + 0.5)
            | (lats >= lats.max() - 0.5)
            | (lons <= lons.min() + 0.5)
            | (lons >= lons.max() - 0.5),
            is_outlier,
        ),
    }

    if coastal is not None:
        report["coastal"] = _subset_report(coastal, is_outlier)

    # Freshest 5% of the day's field: the specific confound worth watching,
    # since Bay of Bengal river output makes low salinity a regional signature.
    fresh_cut = float(np.percentile(salinity, 5))
    report["low_salinity"] = {
        "threshold_psu": round(fresh_cut, 3),
        **_subset_report(salinity <= fresh_cut, is_outlier),
    }

    flagged = is_outlier.sum()
    report["outlier_salinity_mean"] = (
        round(float(salinity[is_outlier].mean()), 3) if flagged else None
    )
    report["field_salinity_mean"] = round(float(salinity.mean()), 3)

    # A plain reading of the numbers above, so a caller does not have to
    # re-derive the same judgement. Thresholds are deliberately blunt.
    notes: list[str] = []
    for name, section in report["basins"].items():
        if section["enrichment"] is not None and section["enrichment"] >= 2.0:
            notes.append(
                f"{name} holds {section['share_of_cells_pct']}% of cells but "
                f"{section['share_of_outliers_pct']}% of outliers "
                f"(enrichment {section['enrichment']}x)"
            )
    for key in ("near_basin_boundary", "domain_edge", "coastal", "low_salinity"):
        section = report.get(key)
        if section and section.get("enrichment") is not None and section["enrichment"] >= 2.0:
            notes.append(
                f"{key} is {section['enrichment']}x over-represented among outliers"
            )
    report["notes"] = notes or ["no subset is over-represented by 2x or more"]
    return report


# --------------------------------------------------------------------------
# Top-level entry points
# --------------------------------------------------------------------------


def detect_anomalies(
    scoring_date: str | date,
    requested_depth: float,
    *,
    config: AnomalyModelConfig | None = None,
    n_estimators: int | None = None,
    contamination: float | None = None,
    max_samples: int | None = None,
    data_dir: Path | None = None,
) -> AnomalyResult:
    """Prepare, fit, score, and return only the cells the forest flagged.

    The whole pipeline for one depth and one day. Repeat calls with the same
    arguments reuse both the depth slices and the fitted model, and return
    byte-identical results -- random_state is fixed and nothing here samples.
    """
    if config is None:
        config = AnomalyModelConfig(
            n_estimators=(
                DEFAULT_N_ESTIMATORS if n_estimators is None else n_estimators
            ),
            contamination=(
                DEFAULT_CONTAMINATION if contamination is None else contamination
            ),
            max_samples=DEFAULT_MAX_SAMPLES if max_samples is None else max_samples,
        )
    config = config.validated()

    dataset = prepare_anomaly_dataset(scoring_date, requested_depth, data_dir=data_dir)
    fitted = fit_anomaly_model(dataset, config, data_dir=data_dir)
    scored = score_field(fitted, dataset)

    flagged = np.flatnonzero(scored.is_outlier)
    # Most anomalous first: lowest raw score leads.
    flagged = flagged[np.argsort(scored.raw_scores[flagged], kind="stable")]

    # Computed after `flagged` is ordered by score, so context[n] belongs to
    # outliers[n].
    contexts = feature_context_for(dataset, fitted, flagged)

    outliers = [
        AnomalyPoint(
            lat=float(dataset.score_latitudes[i]),
            lon=float(dataset.score_longitudes[i]),
            temperature=float(dataset.X_score[i, 0]),
            salinity=float(dataset.X_score[i, 1]),
            current_speed=float(dataset.X_score[i, 2]),
            raw_score=float(scored.raw_scores[i]),
            decision_function=float(scored.decision_scores[i]),
            anomaly_percentile=float(scored.percentiles[i]),
            severity=severity_for(float(scored.percentiles[i])),
            feature_context=contexts[position],
        )
        for position, i in enumerate(flagged)
    ]

    geography = geographic_summary(
        dataset,
        scored.is_outlier,
        coastal_flags(dataset.scoring_date, requested_depth, data_dir),
    )

    return AnomalyResult(
        model_name=MODEL_NAME,
        feature_names=dataset.feature_names,
        requested_depth=dataset.requested_depth,
        actual_depth=dataset.actual_depth,
        scoring_date=dataset.scoring_date,
        training_dates=list(dataset.training_dates),
        training_strategy=dataset.training_strategy,
        excluded_scoring_date=dataset.excluded_scoring_date,
        training_sample_count=dataset.training_sample_count,
        scoring_sample_count=dataset.scoring_sample_count,
        n_estimators=config.n_estimators,
        max_samples=fitted.effective_max_samples,
        contamination=config.contamination,
        random_state=config.random_state,
        outlier_count=len(outliers),
        outlier_percentage=round(
            100.0 * len(outliers) / dataset.scoring_sample_count, 4
        ),
        outliers=outliers,
        dropped_dates=list(dataset.dropped_dates),
        geography=geography,
        fit_seconds=fitted.fit_seconds,
        score_seconds=scored.score_seconds,
    )


def score_all_points(
    scoring_date: str | date,
    requested_depth: float,
    *,
    config: AnomalyModelConfig | None = None,
    data_dir: Path | None = None,
) -> list[dict]:
    """Debug only: every scored cell, including the ones that were not flagged.

    detect_anomalies() deliberately drops these -- ~65k rows is not something to
    put on a wire. This exists for inspecting the score distribution by hand.
    """
    dataset = prepare_anomaly_dataset(scoring_date, requested_depth, data_dir=data_dir)
    fitted = fit_anomaly_model(dataset, config, data_dir=data_dir)
    scored = score_field(fitted, dataset)

    return [
        {
            "lat": float(dataset.score_latitudes[i]),
            "lon": float(dataset.score_longitudes[i]),
            "temperature": float(dataset.X_score[i, 0]),
            "salinity": float(dataset.X_score[i, 1]),
            "current_speed": float(dataset.X_score[i, 2]),
            "raw_score": float(scored.raw_scores[i]),
            "decision_function": float(scored.decision_scores[i]),
            "anomaly_percentile": float(scored.percentiles[i]),
            "severity": severity_for(float(scored.percentiles[i])),
            "is_outlier": bool(scored.is_outlier[i]),
        }
        for i in range(dataset.scoring_sample_count)
    ]
