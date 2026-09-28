"""
Shared pieces of the animal-telemetry layer.

A provider turns one published tracking dataset into a table of normalised GPS
fixes (NORMALISED_COLUMNS) plus per-animal deployment details. Everything the
API serves -- the animal list, one animal, one track -- is built here from that
table, so the router never sees a provider's own column names.

Rules every provider inherits:

  * Fixes are recorded positions, served with the timestamp the source gives,
    in UTC. Nothing is interpolated, resampled or invented between them.
  * depth_m stays null when the source records no depth. It is never 0 m.
  * No fix is dropped for looking wrong. Suspicious fixes are kept and flagged
    (movement_qc_flag), with the reason, so the reader can decide.
  * Data is historical and labelled so: never "live", "real-time" or "current".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATA_STATUS_HISTORICAL = "historical"

RECORDED_TRACK_NOTE = (
    "Historical telemetry. Each position is a recorded GPS fix; lines between them "
    "only connect recorded positions in time order. They are not the exact path the "
    "animal swam, which is unknown between fixes."
)
DEPTH_NOTE = (
    "The source records no depth for these fixes, so depth_m is null. It is unknown, "
    "not 0 m."
)

# Movement QC ----------------------------------------------------------------
#
# The apparent displacement rate is the great-circle distance between two
# consecutive recorded fixes divided by the time between them. It is not a
# swimming speed: the animal's real path between fixes is unknown, and ocean
# currents carry it as well.
#
# 5 km/h is a screening threshold, not a biological limit. It sits well above
# typical green turtle travel rates, and in the Chagos dataset fewer than 1% of
# consecutive-fix rates exceed it. A rate above it can mean that either fix is in
# error, or that a strong current was moving the animal. Flagged fixes are kept.
MOVEMENT_QC_THRESHOLD_KMH = 5.0
MOVEMENT_QC_METHOD = (
    "Apparent displacement rate between recorded fixes: great-circle (haversine) "
    "distance from the previous recorded fix of the same animal, divided by the time "
    "between them. A fix is flagged 'suspect' when that rate exceeds "
    f"{MOVEMENT_QC_THRESHOLD_KMH:g} km/h. This is a QC heuristic, not a swimming "
    "speed and not a biological limit: a flag means one of the two fixes may be in "
    "error, or a current was carrying the animal. Flagged fixes are kept, never removed."
)

EARTH_RADIUS_KM = 6371.0088

# The table every provider produces, one row per recorded fix.
NORMALISED_COLUMNS = (
    "animal_id",
    "species_common",
    "species_scientific",
    "timestamp",            # pandas Timestamp, UTC
    "latitude",
    "longitude",
    "depth_m",              # NaN when unrecorded; served as null
    "source",
    "provider",
    "dataset_id",
    "study_name",
    "tracking_method",
    "data_status",
    "satellite_count",      # NaN when the source has none
    "gps_residual",         # the source's own quality indicator, unitless
    "source_event_id",
    "source_outlier",       # True when the source itself marked the fix as an outlier
)


class AnimalDataUnavailableError(RuntimeError):
    """No verified copy of a dataset, and it could not be downloaded."""


class AnimalNotFoundError(LookupError):
    """No animal with that id in any loaded dataset."""


class FixNotFoundError(LookupError):
    """The animal exists, but no recorded fix matches the request."""


class AmbiguousFixError(ValueError):
    """More than one recorded fix matches; the caller must name one."""


@dataclass(frozen=True)
class DatasetInfo:
    """Provenance for one published dataset. Served with every response."""

    dataset_id: str
    title: str
    study_name: str
    doi: str
    provider: str
    authors: tuple[str, ...]
    landing_page: str
    licence: str
    licence_url: str
    citation: str
    related_publications: tuple[str, ...]
    tracking_method: str
    timestamp_note: str
    data_status: str = DATA_STATUS_HISTORICAL

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}"

    @property
    def source(self) -> str:
        return f"{self.provider}, doi:{self.doi}"

    def as_dict(self, retrieved_at: str | None = None) -> dict:
        return {
            "provider": self.provider,
            "dataset_id": self.dataset_id,
            "dataset_title": self.title,
            "study_name": self.study_name,
            "doi": self.doi,
            "doi_url": self.doi_url,
            "landing_page": self.landing_page,
            "authors": list(self.authors),
            "licence": self.licence,
            "licence_url": self.licence_url,
            "citation": self.citation,
            "related_publications": list(self.related_publications),
            "tracking_method": self.tracking_method,
            "data_status": self.data_status,
            "timestamp_note": self.timestamp_note,
            "retrieved_at": retrieved_at,
        }


@dataclass
class LoadReport:
    """What happened to the source rows on the way in. Served, never hidden."""

    source_rows: int = 0
    loaded_fixes: int = 0
    excluded: dict[str, int] = field(default_factory=dict)


def as_utc(moment) -> pd.Timestamp:
    """A pandas Timestamp in UTC. A moment with no timezone is taken as UTC."""
    stamp = pd.Timestamp(moment)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def iso_utc(moment) -> str | None:
    if moment is None or pd.isna(moment):
        return None
    return as_utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_movement_qc(fixes: pd.DataFrame) -> pd.DataFrame:
    """Add apparent displacement rate and the movement QC flag to every fix.

    Computed over each animal's whole track, before any time filtering, so the
    first fix of a requested window is still compared with the fix before it.
    """
    fixes = fixes.sort_values(["animal_id", "timestamp", "source_event_id"], kind="mergesort").reset_index(drop=True)
    same_animal = fixes["animal_id"].eq(fixes["animal_id"].shift())
    distance = haversine_km(fixes["latitude"].shift(), fixes["longitude"].shift(),
                            fixes["latitude"], fixes["longitude"])
    hours = (fixes["timestamp"] - fixes["timestamp"].shift()).dt.total_seconds() / 3600.0

    distance = distance.where(same_animal)
    hours = hours.where(same_animal)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (distance / hours).where(hours > 0)

    flag = pd.Series("ok", index=fixes.index, dtype=object)
    reason = pd.Series(None, index=fixes.index, dtype=object)

    first = ~same_animal
    flag.loc[first] = "not_assessed"
    reason.loc[first] = "First recorded fix of this animal: there is no earlier fix to compare with."

    simultaneous = same_animal & hours.eq(0)
    flag.loc[simultaneous] = "suspect"
    reason.loc[simultaneous] = "Same timestamp as the previous recorded fix; a displacement rate cannot be computed."

    fast = same_animal & (rate > MOVEMENT_QC_THRESHOLD_KMH)
    flag.loc[fast] = "suspect"
    reason.loc[fast] = [
        f"Apparent displacement rate from the previous recorded fix is {r:.1f} km/h "
        f"({d:.2f} km in {h:.2f} h), above the {MOVEMENT_QC_THRESHOLD_KMH:g} km/h QC threshold. "
        "One of the two fixes may be in error, or a current was carrying the animal."
        for r, d, h in zip(rate[fast], distance[fast], hours[fast])
    ]

    outlier = fixes["source_outlier"].fillna(False).astype(bool)
    flag.loc[outlier] = "suspect"
    reason.loc[outlier] = [
        (f"{existing} " if existing else "") + "Marked as an outlier in the source dataset."
        for existing in reason[outlier].fillna("")
    ]

    return fixes.assign(
        distance_from_previous_km=distance,
        hours_since_previous=hours,
        apparent_displacement_rate_kmh=rate,
        movement_qc_flag=flag,
        movement_qc_reason=reason,
    )


def _number(value, digits: int | None = None):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return round(value, digits) if digits is not None else value


def _integer(value):
    return None if value is None or pd.isna(value) else int(value)


class AnimalTrackProvider(ABC):
    """One published tracking dataset.

    Subclasses implement load(); everything served is derived here from the
    normalised table, so every provider answers the same questions the same way.
    """

    dataset: DatasetInfo

    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, dict[str, dict], LoadReport, str | None]:
        """(normalised fixes with movement QC, deployments by animal id, load report, retrieved_at).

        Raises AnimalDataUnavailableError when there is no verified copy.
        """

    # -- summaries ---------------------------------------------------------

    def _summary(self, animal_id: str, track: pd.DataFrame) -> dict:
        first = track.iloc[0]
        return {
            "animal_id": animal_id,
            "species_common": first["species_common"],
            "species_scientific": first["species_scientific"],
            "first_recorded": iso_utc(track["timestamp"].iloc[0]),
            "last_recorded": iso_utc(track["timestamp"].iloc[-1]),
            "fix_count": int(len(track)),
            "tracking_method": self.dataset.tracking_method,
            "source": self.dataset.source,
            "dataset_id": self.dataset.dataset_id,
            # The same species can be tracked by more than one study, so the
            # study is named on every animal rather than inferred from species.
            "study_name": self.dataset.study_name,
            "doi": self.dataset.doi,
            "data_status": self.dataset.data_status,
        }

    def list_animals(self, species: str | None = None) -> list[dict]:
        fixes, _, _, _ = self.load()
        animals = []
        for animal_id, track in fixes.groupby("animal_id", sort=True):
            if species and not _species_matches(track.iloc[0], species):
                continue
            animals.append(self._summary(animal_id, track))
        return animals

    def has_animal(self, animal_id: str) -> bool:
        fixes, _, _, _ = self.load()
        return bool((fixes["animal_id"] == animal_id).any())

    def get_animal(self, animal_id: str) -> dict:
        fixes, deployments, report, retrieved_at = self.load()
        track = fixes[fixes["animal_id"] == animal_id]
        if track.empty:
            raise AnimalNotFoundError(animal_id)
        last = track.iloc[-1]
        flagged = int((track["movement_qc_flag"] == "suspect").sum())
        return {
            **self._summary(animal_id, track),
            "latest_recorded_position": {
                "timestamp": iso_utc(last["timestamp"]),
                "lat": _number(last["latitude"]),
                "lon": _number(last["longitude"]),
                "label": "Latest recorded position (historical): where the last GPS fix was recorded, not where the animal is now.",
            },
            "deployment": deployments.get(animal_id),
            "quality_metadata": {
                "satellite_count_available": bool(track["satellite_count"].notna().any()),
                "gps_residual_available": bool(track["gps_residual"].notna().any()),
                "gps_residual_meaning": self.residual_meaning(),
                "location_error_m": None,
                "location_error_note": (
                    "The source gives no location error in metres, so none is stated. "
                    "Satellite count and residual are passed through as published."
                ),
                "depth_available": bool(track["depth_m"].notna().any()),
                "movement_qc_threshold_kmh": MOVEMENT_QC_THRESHOLD_KMH,
                "movement_qc_flagged_fixes": flagged,
            },
            "recorded_track_note": RECORDED_TRACK_NOTE,
            "provenance": self.dataset.as_dict(retrieved_at),
            "load_report": {"source_rows": report.source_rows, "loaded_fixes": report.loaded_fixes,
                            "excluded": dict(report.excluded)},
        }

    def get_track(self, animal_id: str, start: datetime | None = None, end: datetime | None = None) -> dict:
        fixes, _, _, retrieved_at = self.load()
        track = fixes[fixes["animal_id"] == animal_id]
        if track.empty:
            raise AnimalNotFoundError(animal_id)
        whole = track
        if start is not None:
            track = track[track["timestamp"] >= as_utc(start)]
        if end is not None:
            track = track[track["timestamp"] <= as_utc(end)]

        positions = [
            {
                "timestamp": iso_utc(row.timestamp),
                "lat": _number(row.latitude),
                "lon": _number(row.longitude),
                "depth_m": _number(row.depth_m),
                "satellite_count": _integer(row.satellite_count),
                "gps_residual": _number(row.gps_residual),
                "apparent_displacement_rate_kmh": _number(row.apparent_displacement_rate_kmh, 3),
                "movement_qc_flag": row.movement_qc_flag,
                "movement_qc_reason": row.movement_qc_reason if isinstance(row.movement_qc_reason, str) else None,
                "source_outlier": bool(row.source_outlier),
                "source_event_id": str(row.source_event_id),
            }
            for row in track.itertuples(index=False)
        ]
        first = whole.iloc[0]
        return {
            "animal_id": animal_id,
            "species_common": first["species_common"],
            "species_scientific": first["species_scientific"],
            "data_status": self.dataset.data_status,
            "tracking_method": self.dataset.tracking_method,
            "requested_start": iso_utc(start) if start is not None else None,
            "requested_end": iso_utc(end) if end is not None else None,
            "first_time": positions[0]["timestamp"] if positions else None,
            "last_time": positions[-1]["timestamp"] if positions else None,
            "position_count": len(positions),
            "recorded_track_note": RECORDED_TRACK_NOTE,
            "depth_note": DEPTH_NOTE,
            "movement_qc": {
                "method": MOVEMENT_QC_METHOD,
                "threshold_kmh": MOVEMENT_QC_THRESHOLD_KMH,
                "flagged_count": sum(p["movement_qc_flag"] == "suspect" for p in positions),
                "not_assessed_count": sum(p["movement_qc_flag"] == "not_assessed" for p in positions),
            },
            "positions": positions,
            "provenance": self.dataset.as_dict(retrieved_at),
        }

    def residual_meaning(self) -> str:
        return "Not provided by this source."


def _species_matches(row: pd.Series, wanted: str) -> bool:
    wanted = wanted.strip().lower()
    return wanted in {str(row["species_scientific"]).lower(), str(row["species_common"]).lower()}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
