"""
Numerical model endpoints.

Serves depth slices of the Copernicus model data that
scripts/download_ocean_data.py has pulled into data/model/, in the shape
main.js needs for its Cesium overlay.
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, Query

from app.models.ocean import (
    CurrentsResponse,
    GridResponse,
    SectionResponse,
    StackResponse,
    TSResponse,
)
from app.models.missions import MissionProvenance, MissionResponse
from app.routers.params import parse_bbox, parse_date, parse_point
from app.services import density, missions as mission_service, ocean_model, region_summary, sections, ts_diagram
from app.services.ocean_model import DatasetNotFoundError
from app.services.sections import SectionError
from app.services.ts_diagram import TSError

router = APIRouter()

# Every field in the dropdown is backed by a downloaded product or derived from
# one; nothing is simulated.
SUPPORTED_VARIABLES = (
    "sst", "salinity", "mld", "sla", "tchp", "o2", "no3", "chl",
    # Model chlorophyll is "chl_model", never "chl": the bare CF name is
    # already taken by the satellite product it must not be confused with.
    "po4", "si", "chl_model", "zos",
    # Photosynthetically active radiation with depth, derived by scripts/build_par.py.
    "par",
    # Derived isotherm depths, the companions to tchp.
    "d26", "d20",
    # Significant wave height, from the Copernicus wave model.
    "wave",
)

# The variables whose downloaded days define the date slider. Mixed layer depth
# and sea level anomaly are served but deliberately left out: sla is a separate
# satellite product on its own delivery schedule, and letting a day it lacks
# drop out of the slider would move the date under every other layer. A day a
# 2D field is missing is handled per request by the usual nearest-day fallback.
DATE_GATING_VARIABLES = ("sst", "salinity")


def _validate_variable(variable: str) -> str:
    key = variable.lower()
    if key in SUPPORTED_VARIABLES:
        return key
    raise HTTPException(
        status_code=400,
        detail=f"unknown variable {variable!r}. Supported: {list(SUPPORTED_VARIABLES)}.",
    )


@router.get("/regions", summary="Surface state of each named sea region on a model day")
def get_region_summary(
    date: str | None = Query(None, description="Model day, YYYY-MM-DD; defaults to the latest downloaded day."),
) -> dict:
    """Bay of Bengal, Northern Bay, Andaman Sea, Arabian Sea, Laccadive Sea and Gulf
    of Mannar: mean and range of sea temperature, salinity, strongest current, wave
    height (midday), mixed layer depth (model); sea level anomaly and chlorophyll
    (satellite, nearest day); and coastal alerts in force now, if Ocean Connect holds
    a copy. Every figure is read off the downloaded fields; land is skipped."""
    if date is None:
        days = ocean_model.available_days("sst")
        if not days:
            raise HTTPException(status_code=404, detail="no model data is downloaded")
        date = str(days[-1])
    else:
        date = str(parse_date(date))
    return region_summary.get_summary(date)


@router.get("/dates", summary="What model data is downloaded")
def list_available_dates() -> dict:
    """Coverage on disk, so callers can pick a date that exists instead of guessing."""
    nc_to_frontend = {v: k for k, v in ocean_model.VARIABLE_TO_NC.items()}

    available = []
    latest: dict[str, str] = {}

    for entry in ocean_model.iter_available_files():
        variable = nc_to_frontend.get(entry["nc_var"])
        if variable is None:
            continue
        available.append(
            {
                "variable": variable,
                "region": entry["region"],
                "start": str(entry["start"]),
                "end": str(entry["end"]),
            }
        )
        end = str(entry["end"])
        if variable not in latest or end > latest[variable]:
            latest[variable] = end

    # The newest day every date-gating variable can serve, so a caller can render
    # the core fields without switching dates.
    scalar_latest = [latest.get(v) for v in DATE_GATING_VARIABLES]
    common = min(scalar_latest) if all(scalar_latest) else None

    # Every individual day servable for ALL supported variables, so a date
    # picker can offer exactly the days that will render rather than a range
    # with holes in it.
    per_variable: dict[str, set] = {}
    for entry in ocean_model.iter_available_files():
        variable = nc_to_frontend.get(entry["nc_var"])
        if variable is None:
            continue
        # The days the file really holds, not the range it was named for.
        per_variable.setdefault(variable, set()).update(entry["days"])

    # Intersect over the SCALAR variables the grid endpoint serves. uo/vo are
    # now in VARIABLE_TO_NC too, but currents have their own endpoint and must
    # not gate the date slider -- comparing counts here would empty it.
    scalar_days = [per_variable.get(v) for v in DATE_GATING_VARIABLES]
    servable = (
        set.intersection(*scalar_days) if all(scalar_days) else set()
    )

    return {
        "available": available,
        "latest": latest,
        "latest_common": common,
        "dates": sorted(str(day) for day in servable),
    }


# Stacked rendering asks for several slices at once. Each is individually
# cached by ocean_model.get_grid(), so a repeat stack costs almost nothing, but
# the count and density still need capping to keep one response sane.
MAX_STACK_SLICES = 8
DEFAULT_STACK_DEPTHS = "0,50,100,200,500,1000"


@router.get("/stack", response_model=StackResponse, summary="Several depth slices at once")
def get_depth_stack(
    date: str = Query(..., description="Day to read, YYYY-MM-DD."),
    variable: str = Query("sst", description='Either "sst" or "salinity".'),
    depths: str = Query(
        DEFAULT_STACK_DEPTHS,
        description=f"Comma-separated depths in metres, at most {MAX_STACK_SLICES}.",
    ),
    bbox: str | None = Query(None, description="Optional clip box 'lonMin,latMin,lonMax,latMax'."),
    stride: int = Query(
        6, ge=2, le=50,
        description="Every Nth cell per axis. Minimum 2: a full-resolution stack is far too large.",
    ),
) -> StackResponse:
    """One request, several depth levels -- each slice reports its own actual depth."""
    variable = _validate_variable(variable)
    if ocean_model.is_surface_variable(variable):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{variable!r} is a 2D field with no depth axis, so it cannot be "
                f"stacked by depth. Request it from /api/model/grid instead."
            ),
        )
    date = parse_date(date)
    box = parse_bbox(bbox) if bbox else None

    try:
        wanted = [float(part) for part in depths.split(",") if part.strip()]
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"depths must be comma-separated numbers, got {depths!r}"
        )
    if not wanted:
        raise HTTPException(status_code=400, detail="depths must list at least one value")
    if len(wanted) > MAX_STACK_SLICES:
        raise HTTPException(
            status_code=400,
            detail=f"at most {MAX_STACK_SLICES} depths per request, got {len(wanted)}",
        )
    if any(d < 0 for d in wanted):
        raise HTTPException(status_code=400, detail="depths cannot be negative")

    requested_date = date
    fallback = False
    if date not in [str(d) for d in ocean_model.available_days(variable)]:
        nearest = ocean_model.find_nearest_available_date(variable, date)
        if nearest is None:
            raise HTTPException(
                status_code=404,
                detail=f"no data for this date/region: no {variable} file exists at all.",
            )
        date = str(nearest)
        fallback = True

    slices = []
    for depth in sorted(set(wanted)):
        points, actual_depth = ocean_model.get_grid(
            variable, date, depth, stride=stride, bbox=box
        )
        if not points:
            continue  # that level is entirely land in this box
        values = [p["value"] for p in points]
        slices.append(
            {
                "depth": depth,
                "actual_depth": round(actual_depth, 3),
                "count": len(points),
                "min_value": round(min(values), 3),
                "max_value": round(max(values), 3),
                "points": points,
            }
        )

    if not slices:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no data for this date/region: no {variable} values at any requested "
                f"depth on {date} (all land, or outside the downloaded grid)."
            ),
        )

    return StackResponse(
        variable=variable,
        date=date,
        requested_date=requested_date,
        fallback=fallback,
        stride=stride,
        bbox=list(box) if box else None,
        total_points=sum(s["count"] for s in slices),
        slices=slices,
    )


@router.get("/section", response_model=SectionResponse, summary="Vertical section along a line")
def get_section(
    date: str = Query(..., description="Day to read, YYYY-MM-DD."),
    start: str = Query(..., description="One end of the transect, 'lat,lon'."),
    end: str = Query(..., description="The other end, 'lat,lon'."),
    variable: str = Query("sst", description="Any 3D field: sst, salinity, o2, no3, po4, si, chl_model, par."),
    samples: int = Query(
        120, ge=2, le=400,
        description="Points along the path, inclusive of both ends.",
    ),
    max_depth: float = Query(
        2000, gt=0, description="Deepest model level to include, in metres."
    ),
) -> SectionResponse:
    """A depth-vs-distance curtain: the model down a line between two points.

    Land, the seafloor and anything outside the downloaded grid come back as
    null rather than as a filled-in value, so the shape of the section is the
    model's own bathymetry.
    """
    variable = _validate_variable(variable)
    if ocean_model.is_surface_variable(variable):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{variable!r} is a 2D field with no depth axis, so it has no vertical "
                f"section. Request it from /api/model/grid instead."
            ),
        )

    date = parse_date(date)
    start_point = parse_point(start, "start")
    end_point = parse_point(end, "end")

    requested_date = date
    fallback = False
    if date not in [str(d) for d in ocean_model.available_days(variable)]:
        nearest = ocean_model.find_nearest_available_date(variable, date)
        if nearest is None:
            raise HTTPException(
                status_code=404,
                detail=f"no data for this date/region: no {variable} file exists at all.",
            )
        date = str(nearest)
        fallback = True

    try:
        result = sections.sample_section(
            variable, date, start_point, end_point,
            samples=samples, max_depth=max_depth,
        )
    except SectionError as exc:
        # A degenerate path or an all-land transect is the caller's input being
        # unanswerable, not a server fault.
        raise HTTPException(status_code=400, detail=str(exc))
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    source, dataset_id = ocean_model.provenance(variable)
    return SectionResponse(
        variable=variable,
        date=date,
        requested_date=requested_date,
        fallback=fallback,
        source=source,
        dataset_id=dataset_id,
        units=ocean_model.VARIABLE_UNITS.get(variable),
        start=[start_point[0], start_point[1]],
        end=[end_point[0], end_point[1]],
        samples=samples,
        **result,
    )


@router.get("/ts", response_model=TSResponse, summary="Temperature-salinity diagram")
def get_ts_diagram(
    date: str = Query(..., description="Day to read, YYYY-MM-DD."),
    point: str | None = Query(None, description="One water column, 'lat,lon'."),
    bbox: str | None = Query(
        None, description="Or a box to scatter over, 'lonMin,latMin,lonMax,latMax'."
    ),
    columns: int = Query(
        36, ge=1, le=ts_diagram.MAX_COLUMNS,
        description="Columns to sample when a bbox is given (squared to a grid).",
    ),
    max_depth: float = Query(2000, gt=0, description="Deepest model level to include."),
) -> TSResponse:
    """Temperature against salinity down the water column, with sigma-theta.

    Temperature and salinity live in separate files, so every point returned is
    paired on the depth coordinate itself at a single grid cell -- never by
    position in two independently land-masked arrays.
    """
    if (point is None) == (bbox is None):
        raise HTTPException(
            status_code=400, detail="give exactly one of 'point' or 'bbox'"
        )

    date = parse_date(date)
    requested_date = date
    fallback = False
    if date not in [str(d) for d in ocean_model.available_days("sst")]:
        nearest = ocean_model.find_nearest_available_date("sst", date)
        if nearest is None:
            raise HTTPException(
                status_code=404, detail="no data for this date/region: no temperature file exists."
            )
        date = str(nearest)
        fallback = True

    try:
        if point is not None:
            lat, lon = parse_point(point, "point")
            result = ts_diagram.column_diagram(date, lat, lon, max_depth=max_depth)
        else:
            box = parse_bbox(bbox)
            result = ts_diagram.region_diagram(
                date, box, columns=columns, max_depth=max_depth
            )
    except TSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    span = ts_diagram.bounds(result["columns"])
    return TSResponse(
        date=date,
        requested_date=requested_date,
        fallback=fallback,
        source="model",
        dataset_ids={
            "temperature": ocean_model.provenance("sst")[1],
            "salinity": ocean_model.provenance("salinity")[1],
        },
        units={"temperature": "degrees_C", "salinity": "PSU", "sigma_theta": "kg/m3"},
        columns=result["columns"],
        column_count=len(result["columns"]),
        point_count=result["point_count"],
        isopycnals=density.sigma_theta_grid(
            tuple(span["salinity_range"]), tuple(span["temperature_range"])
        ),
        **span,
    )


@router.get("/currents", response_model=CurrentsResponse, summary="Current vectors")
def get_currents(
    date: str = Query(..., description="Day to read, YYYY-MM-DD."),
    depth: float = Query(0, ge=0, description="Depth in metres; snaps to the nearest level."),
    bbox: str | None = Query(
        None, description="Optional clip box 'lonMin,latMin,lonMax,latMax'."
    ),
    stride: int = Query(
        8, ge=1, le=60,
        description="Every Nth grid cell per axis. Higher = fewer, sparser arrows.",
    ),
) -> CurrentsResponse:
    """Real eastward/northward model velocities, downsampled for drawing."""
    date = parse_date(date)
    box = parse_bbox(bbox) if bbox else None

    requested_date = date
    fallback = False

    try:
        vectors, actual_depth = ocean_model.get_current_vectors(
            date, depth, stride=stride, bbox=box
        )
    except DatasetNotFoundError:
        nearest = ocean_model.find_nearest_available_date("uo", date)
        if nearest is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "no data for this date/region: no current files are downloaded. "
                    "Get them with: python scripts/download_ocean_data.py "
                    f"--variables uo vo --start-date {date} --end-date {date}"
                ),
            )
        date = str(nearest)
        fallback = True
        vectors, actual_depth = ocean_model.get_current_vectors(
            date, depth, stride=stride, bbox=box
        )

    if not vectors:
        where = f" within bbox {bbox}" if bbox else ""
        raise HTTPException(
            status_code=404,
            detail=(
                f"no data for this date/region: no current vectors on {date} at "
                f"{depth:g} m{where} (all land, or outside the downloaded grid)."
            ),
        )

    return CurrentsResponse(
        depth=depth,
        actual_depth=round(actual_depth, 3),
        date=date,
        requested_date=requested_date,
        fallback=fallback,
        model_time=f"{date}T{ocean_model.DAILY_MEAN_CENTRE_HOUR:02d}:00:00",
        stride=stride,
        count=len(vectors),
        max_speed=round(max(v["speed"] for v in vectors), 4),
        vectors=vectors,
    )


@router.get("/grid", response_model=GridResponse, summary="Model grid at a depth")
def get_model_grid(
    variable: str = Query(
        "sst",
        description=(
            'One of "sst", "salinity", "mld" (mixed layer depth, model), '
            '"sla" (sea level anomaly, satellite), "tchp" (tropical cyclone heat '
            'potential, derived from model temperature), "o2" or "no3" (dissolved '
            'oxygen and nitrate, biogeochemistry model) or "chl" (chlorophyll-a, '
            'satellite ocean colour).'
        ),
    ),
    depth: float = Query(
        0, ge=0,
        description=(
            "Depth in metres; snaps to the nearest model level. Ignored for the 2D "
            "fields mld, sla and tchp."
        ),
    ),
    date: str = Query(..., description="Day to read, YYYY-MM-DD."),
    bbox: str | None = Query(
        None,
        description="Optional clip box 'lonMin,latMin,lonMax,latMax' (west,south,east,north).",
    ),
    stride: int = Query(
        1, ge=1, le=50, description="Return every Nth cell per axis; 1 returns the full grid."
    ),
) -> GridResponse:
    """Return one depth slice as a flat list of ocean points, land excluded."""
    variable = _validate_variable(variable)
    date = parse_date(date)
    box = parse_bbox(bbox) if bbox else None

    requested_date = date
    fallback = False

    try:
        points, actual_depth = ocean_model.get_grid(
            variable, date, depth, stride=stride, bbox=box
        )
    except DatasetNotFoundError:
        # Nothing covers this day. Serve the closest downloaded one and say so,
        # so the caller shows a message rather than an empty map.
        nearest = ocean_model.find_nearest_available_date(variable, date)
        if nearest is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no data for this date/region: no {variable} file exists at all. "
                    f"Download it with: python scripts/download_ocean_data.py "
                    f"--variables {ocean_model.resolve_nc_variable(variable)} "
                    f"--start-date {date} --end-date {date}"
                ),
            )
        date = str(nearest)
        fallback = True
        points, actual_depth = ocean_model.get_grid(
            variable, date, depth, stride=stride, bbox=box
        )

    if not points:
        where = f" within bbox {bbox}" if bbox else ""
        raise HTTPException(
            status_code=404,
            detail=(
                f"no data for this date/region: {variable} on {date} at {depth:g} m "
                f"has no ocean points{where} (all land, or the box lies outside the "
                f"downloaded grid)."
            ),
        )

    source, dataset_id = ocean_model.provenance(variable)
    # For a gap-filled satellite product, say how much of this view was filled in.
    interpolated = ocean_model.interpolated_fraction(variable, date, stride=stride, bbox=box)

    return GridResponse(
        variable=variable,
        depth=depth,
        # The grid's 40 levels are uneven, so report which one was actually used.
        # None for a 2D field, which has no levels at all.
        actual_depth=None if actual_depth is None else round(actual_depth, 3),
        date=date,
        requested_date=requested_date,
        fallback=fallback,
        source=source,
        dataset_id=dataset_id,
        units=ocean_model.VARIABLE_UNITS.get(variable),
        interpolated_fraction=interpolated,
        points=points,
    )


GLIDER_ROUTE_NOTE = (
    "planned -- an operator's proposed route and dive profile, not a recorded track. "
    "No glider reports in this basin."
)
NO_VEHICLE_NOTE = (
    "No battery, speed or vehicle state is shown: there is no glider here to report it. "
    "Only the ocean along the route is real."
)


@router.get("/mission", response_model=MissionResponse,
            summary="The model along a planned glider route")
def get_mission(
    date: str = Query(..., description="Day to read, YYYY-MM-DD."),
    waypoints: str = Query(
        ...,
        description="Route as 'lat,lon;lat,lon;...', two to twelve points.",
        examples=["11,82;13,85;15,87"],
    ),
    variables: str = Query(
        "sst,salinity",
        description="Comma-separated 3D fields to sample along the route.",
    ),
    max_depth: float = Query(500.0, gt=0, description="Deepest the glider flies, in metres."),
    cycles: int = Query(4, ge=1, le=20, description="Surface-to-depth dives along the route."),
    samples: int = Query(160, ge=2, le=400, description="Points along the route."),
) -> MissionResponse:
    """What the ocean model holds along a route someone proposes to fly.

    The route and its sawtooth are a **plan**; every value is the **model** for
    that day. Land, the seafloor and anything outside the downloaded grid come
    back as null rather than a filled-in number, and each value reports the
    model level it was actually read from.
    """
    day = parse_date(date)
    names = [_validate_variable(v) for v in variables.split(",") if v.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="name at least one variable")

    points: list[tuple[float, float]] = []
    for index, chunk in enumerate(w for w in waypoints.split(";") if w.strip()):
        point = parse_point(chunk, f"waypoint {index + 1}")
        points.append((point[0], point[1]))

    try:
        result = mission_service.sample_mission(
            names, day, points,
            samples=samples, max_depth=max_depth, cycles=cycles,
        )
    except mission_service.MissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return MissionResponse(
        date=day,
        provenance=MissionProvenance(
            route_type=GLIDER_ROUTE_NOTE,
            value_source="model",
            dataset_note=f"Copernicus analysis-forecast model fields for {day}.",
            depth_rule=("snapped to the nearest model level no deeper than max_depth; "
                        "never interpolated between levels"),
            no_vehicle_note=NO_VEHICLE_NOTE,
        ),
        **result,
    )
