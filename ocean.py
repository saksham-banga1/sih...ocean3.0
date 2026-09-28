"""Pydantic schemas for the ocean model endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GridPoint(BaseModel):
    """One ocean cell at the selected depth. Land cells are never emitted."""

    lat: float = Field(..., examples=[14.2])
    lon: float = Field(..., examples=[72.8])
    value: float = Field(..., examples=[23.4])


class GridResponse(BaseModel):
    """A single depth slice of the numerical model, ready for the Cesium overlay."""

    variable: str = Field(..., examples=["sst"], description="The variable requested.")
    depth: float = Field(..., examples=[100], description="The depth requested, in metres.")
    actual_depth: float | None = Field(
        ...,
        examples=[92.326],
        description=(
            "The model level the values actually come from. The grid has 40 uneven "
            "levels, so a requested depth snaps to the nearest one -- 100 m lands on "
            "92.326 m. Show this in the UI rather than the requested depth. "
            "null for the 2D fields (mld, sla, tchp), which have no vertical axis."
        ),
    )
    date: str = Field(
        ..., examples=["2026-09-26"], description="The day these values actually come from."
    )
    requested_date: str = Field(
        ...,
        examples=["2026-09-26"],
        description="The day that was asked for. Differs from `date` when `fallback` is true.",
    )
    fallback: bool = Field(
        False,
        description=(
            "True when the requested day had no downloaded data and the nearest "
            "available day was served instead. Surface this to the user."
        ),
    )
    source: str = Field(
        "model",
        examples=["model"],
        description=(
            "'model' for Copernicus numerical model output, 'satellite' for an "
            "observed product (sea level anomaly), 'derived' for a quantity "
            "calculated from model output (tropical cyclone heat potential, from "
            "model temperature). Label the layer accordingly."
        ),
    )
    dataset_id: str | None = Field(
        None,
        examples=["cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"],
        description="Copernicus dataset the values come from.",
    )
    units: str | None = Field(None, examples=["degrees_C"], description="Units of `value`.")
    interpolated_fraction: float | None = Field(
        None,
        examples=[0.473],
        description=(
            "For a gap-filled satellite product, the share of the water cells in this "
            "view whose values were interpolated under cloud rather than observed. "
            "null for every product without such flags. Show it: in the monsoon this "
            "reaches half the chlorophyll map."
        ),
    )
    points: list[GridPoint]


class CurrentVector(BaseModel):
    """One model current vector. u/v are the model's own velocities, in m/s."""

    lat: float = Field(..., examples=[14.25])
    lon: float = Field(..., examples=[88.5])
    u: float = Field(..., examples=[0.184], description="Eastward velocity (uo), m/s.")
    v: float = Field(..., examples=[-0.072], description="Northward velocity (vo), m/s.")
    speed: float = Field(..., examples=[0.1976], description="sqrt(u^2 + v^2), m/s.")


class CurrentsResponse(BaseModel):
    """Downsampled current field for one depth and day."""

    variable: str = Field("currents", examples=["currents"])
    depth: float = Field(..., examples=[0])
    actual_depth: float = Field(
        ..., examples=[0.494], description="Model level the vectors came from."
    )
    date: str = Field(..., examples=["2026-09-26"], description="Day the vectors come from.")
    requested_date: str = Field(..., examples=["2026-09-26"])
    fallback: bool = Field(False, description="True when the nearest available day was used.")
    model_time: str = Field(
        ..., examples=["2026-09-26T12:00:00"], description="Instant of the daily mean."
    )
    stride: int = Field(..., examples=[8], description="Every Nth grid cell, per axis.")
    count: int = Field(..., examples=[412])
    max_speed: float = Field(..., examples=[1.42], description="Fastest vector, m/s.")
    dataset_id: str = Field(
        "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        description="Copernicus dataset the components come from.",
    )
    vectors: list[CurrentVector]


class DepthSlice(BaseModel):
    """One horizontal slice of the model at a single depth level."""

    depth: float = Field(..., examples=[100], description="Depth that was requested.")
    actual_depth: float = Field(
        ...,
        examples=[92.326],
        description="Model level the values actually come from; levels are uneven.",
    )
    count: int = Field(..., examples=[2082])
    min_value: float = Field(..., examples=[15.79])
    max_value: float = Field(..., examples=[28.78])
    points: list[GridPoint]


class StackResponse(BaseModel):
    """Several depth slices in one response, for stacked-layer rendering."""

    variable: str = Field(..., examples=["sst"])
    date: str = Field(..., examples=["2026-09-26"])
    requested_date: str = Field(..., examples=["2026-09-26"])
    fallback: bool = Field(False, description="True when the nearest available day was used.")
    stride: int = Field(..., examples=[6], description="Every Nth grid cell per axis.")
    bbox: list[float] | None = Field(None, examples=[[62.0, 4.0, 96.0, 24.0]])
    total_points: int = Field(..., examples=[12492], description="Across all slices.")
    slices: list[DepthSlice]


class SectionResponse(BaseModel):
    """The model along a transect: a depth-vs-distance curtain of one variable."""

    variable: str = Field(..., examples=["sst"])
    date: str = Field(..., examples=["2026-09-26"], description="The day these values come from.")
    requested_date: str = Field(..., examples=["2026-09-26"])
    fallback: bool = Field(False, description="True when the nearest available day was used.")
    source: str = Field(
        ..., examples=["model"], description="'model', 'satellite' or 'derived'."
    )
    dataset_id: str | None = Field(None, examples=["cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"])
    units: str | None = Field(None, examples=["degrees_C"])
    start: list[float] = Field(..., examples=[[16.0, 62.0]], description="[lat, lon] of the A end.")
    end: list[float] = Field(..., examples=[[16.0, 72.0]], description="[lat, lon] of the B end.")
    samples: int = Field(..., examples=[120], description="Points along the path, inclusive.")
    total_distance_km: float = Field(
        ..., examples=[1069.2], description="Great-circle length of the transect."
    )
    bearing_deg: float = Field(
        ..., examples=[89.6], description="Initial compass bearing from A to B."
    )
    depths: list[float] = Field(
        ...,
        examples=[[0.494, 1.541, 2.646]],
        description=(
            "The model's own levels, uneven and unresampled. One row of `values` "
            "per level, in this order."
        ),
    )
    distances_km: list[float] = Field(
        ..., description="Distance from the A end for each column of `values`."
    )
    latitudes: list[float] = Field(..., description="Latitude of each column.")
    longitudes: list[float] = Field(..., description="Longitude of each column.")
    values: list[list[float | None]] = Field(
        ...,
        description=(
            "values[depth][sample]. null where the model has no value -- land, "
            "below the seafloor, or outside the downloaded grid. Nulls are the "
            "model's own coastline and bathymetry, not a mask applied here."
        ),
    )
    water_fraction: float = Field(
        ...,
        examples=[0.612],
        description="Share of the curtain that carries a value rather than null.",
    )
    min_value: float = Field(..., examples=[5.98])
    max_value: float = Field(..., examples=[29.41])


class TSLevel(BaseModel):
    """One model level as a temperature-salinity point."""

    depth: float = Field(..., examples=[155.851], description="Model level, metres.")
    temperature: float = Field(..., examples=[21.429], description="Potential temperature, °C.")
    salinity: float = Field(..., examples=[36.0054], description="Practical salinity.")
    sigma_theta: float = Field(
        ...,
        examples=[25.4312],
        description="Potential density anomaly, kg/m³ (EOS-80 at one atmosphere).",
    )


class TSColumn(BaseModel):
    """One water column, at the model cell nearest the position asked for."""

    lat: float = Field(..., examples=[16.0])
    lon: float = Field(..., examples=[66.0])
    levels: list[TSLevel]


class TSResponse(BaseModel):
    """A temperature-salinity diagram: the plot that separates water masses."""

    date: str = Field(..., examples=["2026-09-26"])
    requested_date: str = Field(..., examples=["2026-09-26"])
    fallback: bool = Field(False, description="True when the nearest available day was used.")
    source: str = Field("model", examples=["model"])
    dataset_ids: dict[str, str | None] = Field(
        ...,
        description="The product each axis comes from; they are two separate files.",
    )
    units: dict[str, str] = Field(
        ..., examples=[{"temperature": "degrees_C", "salinity": "PSU", "sigma_theta": "kg/m3"}]
    )
    columns: list[TSColumn]
    column_count: int = Field(..., examples=[1])
    point_count: int = Field(..., examples=[40], description="Total T-S points returned.")
    temperature_range: list[float] = Field(..., examples=[[2.6, 28.4]])
    salinity_range: list[float] = Field(..., examples=[[34.9, 36.6]])
    sigma_range: list[float] = Field(..., examples=[[22.9, 27.7]])
    points_outside: int = Field(
        0,
        examples=[12],
        description=(
            "Points whose temperature or salinity falls outside the ranges above, "
            "and which a plot drawn to those axes therefore does not show. Once "
            "there are enough points the axes come from the 1st and 99th "
            "percentiles, so a few extremes -- a river plume beside open ocean -- "
            "cannot crush everything else into a corner. Nothing is removed from "
            "`columns`; only the view is narrowed, and this says by how much."
        ),
    )
    isopycnals: dict = Field(
        ...,
        description=(
            "Sigma-theta over the plot's own T-S rectangle, for drawing density "
            "contours: {salinities, temperatures, sigma[t][s]}. Computed on the "
            "server so the equation of state has one implementation, not two."
        ),
    )
