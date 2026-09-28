"""
Historical marine-animal telemetry: tagged animals and their recorded GPS fixes.

Served from published, DOI-versioned tracking datasets (see
app/services/animals/). Everything here is historical: a track is a sequence of
recorded positions, and the lines between them are not the path the animal
swam. Nothing is live, and nothing is generated when data is missing.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Path, Query

from app.models.animals import (
    AnimalDetailResponse,
    AnimalEnvironmentResponse,
    AnimalListResponse,
    AnimalTrackEnvironmentResponse,
    AnimalTrackResponse,
)
from app.routers.params import parse_timestamp
from app.services import animals as animal_service
from app.services.animals import (
    AmbiguousFixError,
    AnimalDataUnavailableError,
    AnimalNotFoundError,
    FixNotFoundError,
    HistoricalContextError,
)

router = APIRouter()

ANIMAL_ID = Path(..., description="The source's individual id, e.g. 52215.", examples=["52215"])


def _not_found(animal_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"no animal {animal_id!r} in the loaded tracking datasets -- see GET /api/animals for the ids",
    )


@router.get("", response_model=AnimalListResponse, summary="Tagged animals with historical telemetry")
def list_animals(
    species: str | None = Query(
        None, description="Only this species: scientific or common name, e.g. 'Chelonia mydas' or 'Green turtle'."
    ),
) -> AnimalListResponse:
    """Every tagged animal, with its recorded period, fix count and source."""
    try:
        return AnimalListResponse(**animal_service.list_animals(species=species))
    except AnimalDataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/{animal_id}", response_model=AnimalDetailResponse, summary="One tagged animal, with provenance")
def get_animal(animal_id: str = ANIMAL_ID) -> AnimalDetailResponse:
    """Summary, deployment details, latest recorded position, data quality and citation."""
    try:
        return AnimalDetailResponse(**animal_service.get_animal(animal_id))
    except AnimalNotFoundError:
        raise _not_found(animal_id)
    except AnimalDataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/{animal_id}/track", response_model=AnimalTrackResponse, summary="An animal's recorded GPS fixes")
def get_track(
    animal_id: str = ANIMAL_ID,
    start_time: str | None = Query(
        None, description="Inclusive, UTC. YYYY-MM-DD (from 00:00) or ISO-8601, e.g. 2018-08-21T00:00:00Z."
    ),
    end_time: str | None = Query(
        None, description="Inclusive, UTC. YYYY-MM-DD (to the end of that day) or ISO-8601."
    ),
) -> AnimalTrackResponse:
    """Recorded fixes in time order, each with its GPS quality fields and movement QC flag."""
    start = parse_timestamp(start_time, "start_time") if start_time else None
    end = parse_timestamp(end_time, "end_time", end_of_day=True) if end_time else None
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start_time must not be after end_time")
    try:
        return AnimalTrackResponse(**animal_service.get_track(animal_id, start, end))
    except AnimalNotFoundError:
        raise _not_found(animal_id)
    except AnimalDataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _context_unavailable(exc: HistoricalContextError) -> HTTPException:
    return HTTPException(status_code=503, detail=f"historical ocean context unavailable: {exc}")


@router.get(
    "/{animal_id}/environment",
    response_model=AnimalEnvironmentResponse,
    summary="Modelled surface-ocean context for one recorded fix",
)
def get_fix_environment(
    animal_id: str = ANIMAL_ID,
    timestamp: str | None = Query(
        None, description="The fix's recorded time, exactly as the track endpoint gives it, e.g. 2018-08-21T01:19:49Z."
    ),
    event_id: str | None = Query(None, description="Or the fix's source_event_id from the track endpoint."),
) -> AnimalEnvironmentResponse:
    """Temperature, salinity and currents from a historical reanalysis, near one recorded fix.

    Surface-ocean context near the recorded position, as a daily mean -- not the
    conditions the animal experienced, whose depth was not recorded. Matched by
    the fix's own time (<= 24 h) to the nearest model water cell (<= 25 km).
    """
    if (timestamp is None) == (event_id is None):
        raise HTTPException(status_code=400, detail="give exactly one of timestamp or event_id")
    wanted = None
    if timestamp is not None:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", timestamp.strip()):
            raise HTTPException(status_code=400, detail="timestamp must name one fix: give its full time, not a date")
        wanted = parse_timestamp(timestamp, "timestamp").strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        return AnimalEnvironmentResponse(**animal_service.get_fix_environment(animal_id, wanted, event_id))
    except AnimalNotFoundError:
        raise _not_found(animal_id)
    except FixNotFoundError:
        raise HTTPException(status_code=404, detail=f"animal {animal_id!r} has no recorded fix at {event_id or wanted}")
    except AmbiguousFixError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except AnimalDataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HistoricalContextError as exc:
        raise _context_unavailable(exc)


@router.get(
    "/{animal_id}/track/environment",
    response_model=AnimalTrackEnvironmentResponse,
    summary="Modelled surface-ocean context for every recorded fix in a period",
)
def get_track_environment(
    animal_id: str = ANIMAL_ID,
    start_time: str | None = Query(None, description="Inclusive, UTC, as for the track endpoint."),
    end_time: str | None = Query(None, description="Inclusive, UTC, as for the track endpoint."),
) -> AnimalTrackEnvironmentResponse:
    """Every fix of the period, in time order, each with its context or the reason there is none."""
    start = parse_timestamp(start_time, "start_time") if start_time else None
    end = parse_timestamp(end_time, "end_time", end_of_day=True) if end_time else None
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start_time must not be after end_time")
    try:
        return AnimalTrackEnvironmentResponse(**animal_service.get_track_environment(animal_id, start, end))
    except AnimalNotFoundError:
        raise _not_found(animal_id)
    except AnimalDataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HistoricalContextError as exc:
        raise _context_unavailable(exc)

