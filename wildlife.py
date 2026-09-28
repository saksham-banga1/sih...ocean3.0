"""
Wildlife Dive: which Indian Ocean megafauna are recorded in each basin, and the
depths they live at.

Two kinds of information, kept apart:

  species reference   a short, cited list of well-studied megafauna -- IUCN Red
                      List category, typical adult length, diet, and the depth
                      band each typically uses, from published diving and tagging
                      studies (sources named per species). Where an animal is
                      drawn in the scene comes from this band, not from any
                      single record.
  OBIS records        real occurrence records from the Ocean Biodiversity
                      Information System (api.obis.org) inside each basin's box:
                      how many, from how many datasets, which years and months,
                      and the distance from shore, water depth and SST that OBIS
                      attaches to each record. A species is listed for a basin
                      only if OBIS holds at least one record of it there.

OBIS records say where an animal was seen, not how deep it swam: almost none of
these carry a depth. "Year-round" and "seasonal" describe the months the records
fall in -- survey effort also varies by season, so they are a reading of the
records, not a residency study.

Basin conditions now (SST, wave height) come from the INCOIS Ocean State
Forecast at each basin's point; seafloor depth from GEBCO (as the Water Column
tab). OBIS summaries are cached on disk for OBIS_REFRESH_SECONDS; if OBIS
cannot be reached, the last cached copy is served and marked stale.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.services import forecast as forecast_service
from app.services import water_column

logger = logging.getLogger(__name__)

OBIS_API = "https://api.obis.org/v3"
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
OBIS_CACHE_DIR = BACKEND_DIR.parent / "data" / "observations" / "obis_cache"
OBIS_REFRESH_SECONDS = 7 * 24 * 3600
OBIS_MAX_RECORDS = 5000          # environmental summaries are taken over at most this many records
YEAR_ROUND_MONTHS = 8            # records in at least this many calendar months: "year-round"
TIMEOUT_SECONDS = 60
MONTH_MS = 31 * 24 * 3600 * 1000

# The basins of the Water Column tab, each with a box for OBIS (lon/lat, west,
# south, east, north). The boxes follow the seas' usual extents; the Arabian Sea
# and Laccadive Sea boxes overlap off south-west India.
BASIN_BOXES = {
    "bay_of_bengal": (80.0, 5.0, 92.0, 22.5),
    "arabian_sea": (60.0, 5.0, 77.0, 25.0),
    "andaman_sea": (92.0, 6.0, 98.5, 16.5),
    "laccadive_sea": (71.5, 5.0, 80.0, 12.5),
}

HOCHSCHEID = "Hochscheid 2014, J. Exp. Mar. Biol. Ecol. 450:118-136 (review of sea turtle diving)"

# Depth bands: [shallow, deep] metres of typical use; `at` is where the scene
# draws the animal, inside that band.
SPECIES: dict[str, dict] = {
    "green_turtle": {
        "common": "Green sea turtle", "scientific": "Chelonia mydas", "group": "turtle", "iucn": "EN",
        "band": [0, 20], "at": 10, "max_dive": None,
        "depth_source": HOCHSCHEID + ": foraging dives in shallow neritic habitat, mostly under 20 m",
        "length": "about 1 m shell", "diet": "Seagrass and algae", "school": 1, "colour": "#34d399",
    },
    "olive_ridley": {
        "common": "Olive ridley turtle", "scientific": "Lepidochelys olivacea", "group": "turtle", "iucn": "VU",
        "band": [0, 100], "at": 35, "max_dive": None,
        "depth_source": HOCHSCHEID + ": oceanic dives mostly within the top 100 m",
        "length": "about 0.7 m shell", "diet": "Crustaceans, jellyfish, fish", "school": 2, "colour": "#a3e635",
    },
    "hawksbill": {
        "common": "Hawksbill turtle", "scientific": "Eretmochelys imbricata", "group": "turtle", "iucn": "CR",
        "band": [0, 20], "at": 16, "max_dive": None,
        "depth_source": HOCHSCHEID + ": coral-reef foraging, mostly under 20 m",
        "length": "about 0.9 m shell", "diet": "Sponges and reef invertebrates", "school": 1, "colour": "#f59e0b",
    },
    "whale_shark": {
        "common": "Whale shark", "scientific": "Rhincodon typus", "group": "shark", "iucn": "EN",
        "band": [0, 100], "at": 70, "max_dive": 1928,
        "depth_source": "Tyminski et al. 2015, PLoS ONE 10:e0120465: most time in the top 100 m; deepest recorded dive 1,928 m",
        "length": "5.5-10 m (up to about 18 m)", "diet": "Plankton, fish eggs, small fish", "school": 1, "colour": "#60a5fa",
    },
    "humpback_dolphin_pacific": {
        "common": "Indo-Pacific humpback dolphin", "scientific": "Sousa chinensis", "group": "dolphin", "iucn": "VU",
        "band": [0, 20], "at": 4, "max_dive": None,
        "depth_source": "Jefferson & Rosenbaum 2014, Mar. Mamm. Sci. 30:1494-1541: coastal, in water mostly under 20 m deep",
        "length": "2-2.8 m", "diet": "Coastal and estuarine fish", "school": 3, "colour": "#fde047",
    },
    "humpback_dolphin_indian": {
        "common": "Indian Ocean humpback dolphin", "scientific": "Sousa plumbea", "group": "dolphin", "iucn": "EN",
        "band": [0, 20], "at": 6, "max_dive": None,
        "depth_source": "Jefferson & Rosenbaum 2014, Mar. Mamm. Sci. 30:1494-1541: coastal, in water mostly under 20 m deep",
        "length": "2-2.8 m", "diet": "Coastal and estuarine fish", "school": 3, "colour": "#facc15",
    },
    "spinner_dolphin": {
        "common": "Spinner dolphin", "scientific": "Stenella longirostris", "group": "dolphin", "iucn": "LC",
        "band": [0, 400], "at": 25, "max_dive": None,
        "depth_source": "Benoit-Bird & Au 2003, Behav. Ecol. Sociobiol. 53:364-373: rest near the surface by day, forage at night on the mesopelagic boundary layer in the upper few hundred metres",
        "length": "1.3-2.1 m", "diet": "Mesopelagic fish, squid, shrimp", "school": 5, "colour": "#e2e8f0",
    },
    "blue_whale": {
        "common": "Blue whale", "scientific": "Balaenoptera musculus", "group": "whale", "iucn": "EN",
        "band": [0, 300], "at": 150, "max_dive": None,
        "depth_source": "Goldbogen et al. 2011, J. Exp. Biol. 214:131-146: lunge-feeding dives on krill, commonly 100-300 m",
        "length": "20-25 m", "diet": "Krill", "school": 1, "colour": "#94a3b8",
    },
    "yellowfin_tuna": {
        "common": "Yellowfin tuna", "scientific": "Thunnus albacares", "group": "tuna", "iucn": "LC",
        "band": [1, 250], "at": 60, "max_dive": None,
        "depth_source": "FishBase (Froese & Pauly, eds.), Thunnus albacares: depth range 1-250 m, mostly in the mixed layer above the thermocline",
        "length": "1-1.5 m (up to about 2 m)", "diet": "Fish, squid, crustaceans", "school": 12, "colour": "#fb923c",
    },
    "sperm_whale": {
        "common": "Sperm whale", "scientific": "Physeter macrocephalus", "group": "sperm", "iucn": "VU",
        "band": [400, 1200], "at": 750, "max_dive": 2000,
        "depth_source": "Watwood et al. 2006, J. Anim. Ecol. 75:814-825: foraging dives mostly 400-1,200 m; dives beyond 2,000 m are known",
        "length": "11-16 m", "diet": "Deep-sea squid", "school": 1, "colour": "#a78bfa",
    },
    "dugong": {
        "common": "Dugong", "scientific": "Dugong dugon", "group": "dugong", "iucn": "VU",
        "band": [0, 10], "at": 6, "max_dive": None,
        "depth_source": "Marsh, O'Shea & Reynolds 2011, Ecology and Conservation of the Sirenia (Cambridge): seagrass meadows, usually under 10 m",
        "length": "about 2.7 m", "diet": "Seagrass", "school": 1, "colour": "#fca5a5",
    },
}

# Well-established facts worth a line, where they apply. Everything else in a
# species' note is read off its OBIS records.
HIGHLIGHTS = {
    ("bay_of_bengal", "olive_ridley"): "Mass nesting (arribadas) at Gahirmatha and Rushikulya, Odisha.",
    ("arabian_sea", "whale_shark"): "Seasonal aggregations along the Saurashtra coast of Gujarat.",
    ("laccadive_sea", "dugong"): "The Gulf of Mannar and Palk Bay seagrass beds hold India's main dugong population.",
    ("andaman_sea", "dugong"): "A small population lives around the Andaman and Nicobar Islands.",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class WildlifeError(RuntimeError):
    pass


class UnknownBasinError(KeyError):
    pass


_lock = threading.Lock()


# --- OBIS -----------------------------------------------------------------------

def _get_json(path: str, params: dict) -> dict:
    """GET an OBIS endpoint. Kept separate so the tests can replace it."""
    url = f"{OBIS_API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "OCEANAO/0.1 (SIH ocean platform)"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def _polygon(box: tuple[float, float, float, float]) -> str:
    w, s, e, n = box
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _obis_summary(box: tuple, scientific: str) -> dict:
    geometry = _polygon(box)
    stats = _get_json("statistics", {"scientificname": scientific, "geometry": geometry})
    records = int(stats.get("records") or 0)
    summary = {
        "records": records,
        "datasets": int(stats.get("datasets") or 0),
        "year_range": stats.get("yearrange"),
        "months": [0] * 12,                 # records dated to within a month, by calendar month
        "records_sampled": 0,
        "median_shore_km": None,
        "median_bathymetry_m": None,
        "median_sst_c": None,
        "latest": None,
        "aphia_id": None,
    }
    if not records:
        return summary
    occ = _get_json("occurrence", {"scientificname": scientific, "geometry": geometry, "size": OBIS_MAX_RECORDS})
    rows = occ.get("results") or []
    shore, bathy, sst, latest = [], [], [], None
    for r in rows:
        mid, start, end = r.get("date_mid"), r.get("date_start"), r.get("date_end")
        if isinstance(mid, (int, float)):
            when = datetime.fromtimestamp(mid / 1000, tz=timezone.utc)
            latest = max(latest, when) if latest else when
            # Only records dated to within a month say which month: one dated
            # "2002/2011" has a mid-point, not a season.
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end - start <= MONTH_MS:
                summary["months"][when.month - 1] += 1
        # OBIS gives distance to shore in metres, negative for points on land
        # (old specimens with coarse localities); those say nothing about the sea.
        if isinstance(r.get("shoredistance"), (int, float)) and r["shoredistance"] >= 0:
            shore.append(r["shoredistance"] / 1000.0)
        # OBIS gives bathymetry as positive depth; negative values are on land.
        if isinstance(r.get("bathymetry"), (int, float)) and r["bathymetry"] > 0:
            bathy.append(float(r["bathymetry"]))
        if isinstance(r.get("sst"), (int, float)):
            sst.append(float(r["sst"]))
        if summary["aphia_id"] is None and r.get("aphiaID"):
            summary["aphia_id"] = int(r["aphiaID"])
    summary.update({
        "records_sampled": len(rows),
        "median_shore_km": _median(shore),
        "median_bathymetry_m": _median(bathy),
        "median_sst_c": _median(sst),
        "latest": latest.date().isoformat() if latest else None,
    })
    return summary


def _cache_file(basin: str) -> Path:
    return OBIS_CACHE_DIR / f"{basin}.json"


def obis_for_basin(basin: str) -> dict:
    """OBIS summaries for every reference species in one basin, cached on disk."""
    path = _cache_file(basin)
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except ValueError:
            cached = None
    if cached and time.time() - cached.get("fetched_epoch", 0) < OBIS_REFRESH_SECONDS and set(cached["species"]) == set(SPECIES):
        return {**cached, "stale": False}
    box = BASIN_BOXES[basin]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {key: pool.submit(_obis_summary, box, sp["scientific"]) for key, sp in SPECIES.items()}
            species = {key: f.result() for key, f in futures.items()}
    except Exception as exc:  # noqa: BLE001
        if cached:
            logger.warning("OBIS unavailable for %s (%s); serving the copy from %s", basin, exc, cached.get("fetched_at"))
            return {**cached, "stale": True}
        raise WildlifeError(f"OBIS is not reachable ({exc}) and no earlier copy exists") from exc
    result = {"species": species, "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "fetched_epoch": time.time(), "box": list(box)}
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    return {**result, "stale": False}


# --- the basin ----------------------------------------------------------------------

def _presence(months: list[int]) -> dict:
    recorded = [MONTHS[i] for i, n in enumerate(months) if n]
    if not recorded:
        return {"label": "UNDATED", "months_recorded": 0, "month_names": []}
    return {
        "label": "YEAR-ROUND" if len(recorded) >= YEAR_ROUND_MONTHS else "SEASONAL",
        "months_recorded": len(recorded),
        "month_names": recorded,
    }


def _note(basin: str, key: str, o: dict) -> str:
    parts = []
    if HIGHLIGHTS.get((basin, key)):
        parts.append(HIGHLIGHTS[(basin, key)])
    yr = o.get("year_range") or []
    span = f", {yr[0]}–{yr[1]}" if len(yr) == 2 else ""
    where = []
    if o.get("median_shore_km") is not None:
        where.append(f"typically {o['median_shore_km']:.0f} km from shore")
    if o.get("median_bathymetry_m") is not None:
        where.append(f"over {o['median_bathymetry_m']:.0f} m of water")
    parts.append(f"{o['records']} OBIS record{'s' if o['records'] != 1 else ''}{span}" + (f", {' '.join(where)}" if where else "") + ".")
    return " ".join(parts)


def get_basin(basin: str) -> dict:
    if basin not in water_column.BASINS:
        raise UnknownBasinError(basin)
    site = water_column.BASINS[basin]
    obis = obis_for_basin(basin)
    try:
        now = forecast_service.conditions_now(basin, site["lat"], site["lon"])
        conditions_error = None
    except Exception as exc:  # noqa: BLE001 -- shown as unavailable, never invented
        now, conditions_error = None, str(exc)

    species, absent = [], []
    for key, sp in SPECIES.items():
        o = obis["species"].get(key) or {"records": 0}
        if not o.get("records"):
            absent.append(sp["common"])
            continue
        species.append({
            "key": key,
            **{k: sp[k] for k in ("common", "scientific", "group", "iucn", "band", "at", "max_dive",
                                  "depth_source", "length", "diet", "school", "colour")},
            "presence": _presence(o["months"]),
            "note": _note(basin, key, o),
            "obis": {k: o.get(k) for k in ("records", "datasets", "year_range", "months", "records_sampled",
                                             "median_shore_km", "median_bathymetry_m", "median_sst_c", "latest", "aphia_id")},
        })
    species.sort(key=lambda s: (s["at"], s["common"]))
    return {
        "basin": basin,
        "name": site["name"],
        "region": site["region"],
        "lat": site["lat"],
        "lon": site["lon"],
        "seafloor_m": water_column.seafloor_depths().get(basin),
        "conditions": now,
        "conditions_error": conditions_error,
        "species": species,
        "not_recorded": absent,
        "obis": {"box": obis["box"], "fetched_at": obis["fetched_at"], "stale": obis.get("stale", False),
                 "year_round_months": YEAR_ROUND_MONTHS, "max_records_summarised": OBIS_MAX_RECORDS},
        "iucn_names": {"CR": "Critically Endangered", "EN": "Endangered", "VU": "Vulnerable", "NT": "Near Threatened", "LC": "Least Concern"},
    }


def list_basins() -> list[dict]:
    depths = water_column.seafloor_depths()
    out = []
    for key, site in water_column.BASINS.items():
        try:
            sst = forecast_service.conditions_now(key, site["lat"], site["lon"]).get("sst_c")
        except Exception:  # noqa: BLE001
            sst = None
        out.append({"basin": key, "name": site["name"], "region": site["region"], "seafloor_m": depths.get(key), "sst_c": sst})
    return out
