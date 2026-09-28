"""Shared query-parameter parsing and validation for the routers."""

from __future__ import annotations

import re
from datetime import datetime, time, timezone

from fastapi import HTTPException


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    """Parse 'lonMin,latMin,lonMax,latMax' into a validated tuple.

    West/south/east/north order, matching Cesium's Rectangle.fromDegrees in
    main.js and app/services/regions.py. Raises HTTPException(400) on bad input.
    """
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"bbox must have 4 comma-separated numbers "
                f"(lonMin,latMin,lonMax,latMax), got {len(parts)}: {raw!r}"
            ),
        )

    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"bbox values must be numbers (lonMin,latMin,lonMax,latMax), got {raw!r}",
        )

    if min_lon >= max_lon:
        raise HTTPException(
            status_code=400,
            detail=f"bbox lonMin ({min_lon:g}) must be less than lonMax ({max_lon:g})",
        )
    if min_lat >= max_lat:
        raise HTTPException(
            status_code=400,
            detail=f"bbox latMin ({min_lat:g}) must be less than latMax ({max_lat:g})",
        )
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise HTTPException(status_code=400, detail="bbox longitudes must be within -180..180")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise HTTPException(status_code=400, detail="bbox latitudes must be within -90..90")

    return min_lon, min_lat, max_lon, max_lat


def parse_date(value: str) -> str:
    """Validate a YYYY-MM-DD date string. Raises HTTPException(400) if malformed."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"invalid date {value!r} -- expected YYYY-MM-DD",
        )
    return value


def parse_point(raw: str, name: str = "point") -> tuple[float, float]:
    """Parse 'lat,lon' into a validated tuple.

    Latitude first, matching how a position reads on screen and in the Argo
    responses. Raises HTTPException(400) on bad input.
    """
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be 'lat,lon', got {len(parts)} values: {raw!r}",
        )
    try:
        lat, lon = (float(p) for p in parts)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"{name} values must be numbers (lat,lon), got {raw!r}"
        )
    if not -90 <= lat <= 90:
        raise HTTPException(
            status_code=400, detail=f"{name} latitude must be within -90..90, got {lat:g}"
        )
    if not -180 <= lon <= 180:
        raise HTTPException(
            status_code=400, detail=f"{name} longitude must be within -180..180, got {lon:g}"
        )
    return lat, lon


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_timestamp(value: str, name: str, end_of_day: bool = False) -> datetime:
    """Parse an ISO-8601 time, or a YYYY-MM-DD date, into an aware UTC datetime.

    A bare date means the whole day: its first instant, or with end_of_day its
    last, so an inclusive end date keeps every fix recorded on that day. A time
    without an offset is taken as UTC. Raises HTTPException(400) if malformed.
    """
    text = value.strip()
    try:
        if _DATE_ONLY.match(text):
            day = datetime.strptime(text, "%Y-%m-%d").date()
            return datetime.combine(day, time.max if end_of_day else time.min, tzinfo=timezone.utc)
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"invalid {name} {value!r} -- expected YYYY-MM-DD or an ISO-8601 time such as 2018-08-21T00:00:00Z",
        )
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
