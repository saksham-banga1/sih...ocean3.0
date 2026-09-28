"""
OCEANAO backend — FastAPI entrypoint.

This is a minimal scaffold: it exposes a /health endpoint and CORS so the
existing frontend (structure.html + main.js, currently opened directly as a
static file or served from a plain local server) can eventually call real
endpoints under app/routers/ instead of the hardcoded mock data in main.js.

Run locally with:
    uvicorn app.main:app --reload
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (animals, anomaly, argo, cyclones, export, forecast, hazards, heatwave, model, ocean_connect,
                         sensors, water_column, wildlife)

load_dotenv()

app = FastAPI(
    title="OCEANAO Backend",
    description="Backend for TarangSetu, Bharat's 3D Ocean Observatory (SIH26067, INCOIS PS67).",
    version="0.1.0",
)

# Origins the frontend can be served from during local development.
# - "http://localhost" / "http://localhost:*" style hosts cover a plain static
#   server (e.g. `python -m http.server`, VS Code Live Server, etc).
# - "null" (the Origin a page opened from file:// sends) is deliberately NOT
#   allowed: with allow_credentials it would let any sandboxed or opaque-origin
#   page call the API. Serve the frontend over http instead.
DEFAULT_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
]

extra_origins = [
    origin.strip()
    for origin in os.getenv("EXTRA_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS + extra_origins,
    # Also match any localhost/127.0.0.1 port, since static dev servers vary.
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Argo provenance rides on headers for /api/sensors, whose body is a
    # bare array; browsers hide custom headers unless they are exposed.
    expose_headers=["X-Argo-Source", "X-Argo-Last-Synced"],
)

app.include_router(model.router, prefix="/api/model", tags=["model"])
app.include_router(argo.router, prefix="/api/argo", tags=["argo"])
app.include_router(sensors.router, prefix="/api/sensors", tags=["sensors"])
app.include_router(sensors.compare_router, prefix="/api/compare", tags=["compare"])
# Experimental: statistical outlier detection over the model fields. Served
# under its own prefix so it is never mistaken for a validated model product.
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["anomaly"])
# Real cyclone best tracks from NOAA IBTrACS; replaces a hardcoded, invented track.
app.include_router(cyclones.router, prefix="/api/cyclones", tags=["cyclones"])
app.include_router(export.router, prefix="/api/export", tags=["export"])
# Historical animal telemetry from published, DOI-versioned tracking datasets.
app.include_router(animals.router, prefix="/api/animals", tags=["animals"])
# Marine heatwaves, detected on the REANALYSIS record -- a different product
# from the analysis-forecast data the model date slider reads.
app.include_router(heatwave.router, prefix="/api/heatwave", tags=["heatwave"])
# 72-hour INCOIS Ocean State Forecast (waves, SST, wind) at INCOIS wave-rider
# buoy sites: the only forward-looking data here -- the Copernicus files are a fixed week.
app.include_router(forecast.router, prefix="/api/forecast", tags=["forecast"])
# Full-depth model columns (surface to seafloor) at four basins, fetched from
# Copernicus Marine on first use: the downloaded grids stop at 2000 m.
app.include_router(water_column.router, prefix="/api/water-column", tags=["water-column"])
# Wildlife Dive: OBIS occurrence records per basin, with cited depth bands per species.
app.include_router(wildlife.router, prefix="/api/wildlife", tags=["wildlife"])
# Hazards: 72-hour outlooks from Copernicus and INCOIS forecasts, and oil-spill drift.
app.include_router(hazards.router, prefix="/api/hazards", tags=["hazards"])


@app.get("/health")
def health() -> dict:
    """Simple liveness check used by the frontend / ops to confirm the API is up."""
    return {"status": "ok"}
# Ocean Connect: Indian Ocean news (GDELT), official ocean events (INCOIS, ITEWC,
# NDMA SACHET) and the INCOIS observing networks, each labelled by source.
app.include_router(ocean_connect.router, prefix="/api/connect", tags=["ocean-connect"])
