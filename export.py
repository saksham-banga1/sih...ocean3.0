"""Exports that need the server: the Excel workbook with charts.

CSV and GeoJSON are built in the browser from the very response the map drew.
A workbook with native charts needs a writer the browser does not have, so it
is built here -- from the same get_grid() call, with the same variable, date,
depth, box and stride the map used, and therefore the same cells.
scripts/test_export_workbook.py holds the two to that, cell for cell.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from app.routers.model import _validate_variable
from app.routers.params import parse_bbox, parse_date
from app.services.ocean_model import DatasetNotFoundError
from app.services.workbook import build_workbook

router = APIRouter()

XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.get("/xlsx", summary="The field on screen as an Excel workbook with charts")
def export_xlsx(
    variable: str = Query(..., description="The field on screen."),
    date: str = Query(..., description="The day the map is showing, YYYY-MM-DD."),
    depth: float = Query(0, ge=0, description="The depth the map requested, in metres."),
    bbox: str | None = Query(None, description="The map's box, 'lonMin,latMin,lonMax,latMax'."),
    stride: int = Query(2, ge=1, le=60, description="The thinning the map used."),
) -> Response:
    """Three sheets: Summary (headline numbers, a pie of value bands, a line of
    the same box across every downloaded day), Data (every cell), and About
    (sources and how each chart was made)."""
    variable = _validate_variable(variable)
    date = parse_date(date)
    box = parse_bbox(bbox) if bbox else None
    try:
        blob, filename, _ = build_workbook(variable, date, depth, stride, box)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return Response(
        content=blob,
        media_type=XLSX_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
