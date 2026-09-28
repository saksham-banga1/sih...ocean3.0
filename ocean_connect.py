"""
Ocean Connect: an Indian Ocean intelligence feed -- news, official ocean
events, and the observing networks -- from public sources, each labelled.

  news          GDELT DOC 2.0 API (api.gdeltproject.org): English-language
                articles from the last three days that mention the Indian Ocean,
                Bay of Bengal, Arabian Sea, Andaman Sea or INCOIS. GDELT indexes
                news; it does not verify it. GDELT asks for at most one request
                every five seconds and throttles heavier queries harder, so
                results are cached and requests spaced. If GDELT refuses and no
                copy is cached, the same search is run on Google News RSS and
                the result is labelled as such.
  events        official bulletins and alerts:
                  * INCOIS coastal alerts -- High Wave, Swell Surge and Ocean
                    Current alerts by coastal district, at INCOIS's own levels
                    (Warning / Alert / Watch), from the feed behind INCOIS's
                    alerts map (incois.gov.in/site/services/hwa.jsp)
                  * INCOIS Indian Tsunami Early Warning Centre (ITEWC): every
                    earthquake it assessed in the past 90 days, with its own
                    tsunami-threat evaluation for India
                    (tsunami.incois.gov.in/itews/DSSProducts/OPR/past90days.json)
                  * NDMA SACHET, India's Common Alerting Protocol feed, which
                    carries the alerts of IMD, INCOIS and state authorities: the
                    coastal and marine ones -- tsunami, high wave, swell surge,
                    storm surge, and sea-state warnings for fishermen.
  observations  INCOIS Ocean Observation Network: the moored, drifting and
                wave-rider buoys and ship weather stations that reported in the
                last three days (positions and times), plus the RAMA and OMNI
                moored-buoy arrays (with INCOIS's own reporting status), the
                tide-gauge network and the HF radar sites, from INCOIS's map
                services; and the Argo floats that surfaced in the last ten days
                (this platform's Argo service). Positions only: the measurements
                themselves are INCOIS's and the Argo data centres' to serve.

Everything is cached (news 20 min, events 10 min, observations 30 min); if a
source cannot be reached, its last copy is served and marked stale.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BACKEND_DIR.parent / "data" / "observations" / "connect_cache"
UA = "Mozilla/5.0 (OCEANAO SIH ocean platform)"
TIMEOUT = 40
SLOW_TIMEOUT = 150                       # INCOIS's mobile-app feeds can take over a minute

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_TERMS = '"Indian Ocean" OR "Bay of Bengal" OR "Arabian Sea" OR "Andaman Sea" OR INCOIS'
GDELT_QUERY = f"({NEWS_TERMS})"          # a sourcelang: operator gets throttled; language is filtered here instead
GDELT_SPACING_S = 6.0
GNEWS_RSS = "https://news.google.com/rss/search"
HWASSA_URL = "https://sarat.incois.gov.in/incoismobileappdata/rest/incois/hwassalatestdata"
CURRENTS_URL = "https://samudra.incois.gov.in/incoismobileappdata/rest/incois/currentslatestdata"
DISTRICTS_URL = "https://sarat.incois.gov.in/incoismobileappdata/rest/incois/districtpolygons"
DISTRICTS_TTL_S = 30 * 86400
IST = timezone(timedelta(hours=5, minutes=30))
ITEWC_URL = "https://tsunami.incois.gov.in/itews/DSSProducts/OPR/past90days.json"
SACHET_RSS = "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml"
OON_URL = "https://incois.gov.in/OON/backend_process.jsp"
WFS = "https://incois.gov.in/geoserver/{ws}/ows?service=WFS&version=1.0.0&request=GetFeature&typeName={layer}&outputFormat=application/json"
HF_RADAR_URL = "https://incois.gov.in/OON/fetchHFRadarBuoyData.jsp"

TTL = {"news": 1200, "events": 600, "observations": 1800}

# Event classes, most serious first, and the words that put an alert in one.
EVENT_CLASSES = [
    ("TSUNAMI", "#ef4444", r"tsunami"),
    ("HIGH WAVE", "#f97316", r"high wave|high waves"),
    ("SWELL SURGE", "#facc15", r"swell surge|swell waves?"),
    ("STORM SURGE", "#3b82f6", r"storm surge"),
    ("OCEAN CURRENT", "#2dd4bf", r"ocean current"),
    ("MARINE WEATHER", "#a78bfa", r"fishermen|rough sea|sea condition|sea state|squall|cyclon|deep depression|"
                                  r"along and off|off .* coast|coast|high sea|gale|sea will be|sea is"),
]
MARINE_SENDERS = re.compile(r"IMD|INCOIS|SDMA", re.I)

TAGS = [("Cyclones & weather", r"cyclon|storm|depression|low.pressure|monsoon|rainfall|heavy rain"),
        ("Coastal hazards", r"high wave|swell|surge|tsunami|earthquake|erosion|coastal flood|kallakadal"),
        ("Fisheries", r"fish|trawl"),
        ("Shipping & security", r"navy|naval|ship|port|vessel|maritime|coast guard|submarine|missile"),
        ("Science & climate", r"climate|warming|heatwave|research|scien|incois|coral|species|marine life|expedition"),
        ("Pollution & spills", r"oil spill|spill|plastic|pollut")]


class ConnectError(RuntimeError):
    pass


_lock = threading.Lock()
_memory: dict[str, tuple[float, dict]] = {}
_building: dict[str, threading.Lock] = {k: threading.Lock() for k in TTL}
_last_gdelt = 0.0


# tsunami.incois.gov.in sends its certificate without the intermediate that
# links it to a trusted root (browsers and curl fetch it themselves; Python does
# not). That public intermediate -- GlobalSign RSA OV SSL CA 2018, issued by
# GlobalSign Root R3, SHA-256 B6:76:FF:A3...:76:4A, from the certificate's own
# CA Issuers URL -- is added to the usual trust store. Verification stays on.
INTERMEDIATES = BACKEND_DIR / "app" / "certs" / "globalsign_rsa_ov_ssl_ca_2018.pem"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    if INTERMEDIATES.exists():
        context.load_verify_locations(cafile=str(INTERMEDIATES))
    return context


_SSL = _ssl_context()


def _fetch(url: str, *, data: bytes | None = None, headers: dict | None = None, timeout: float = TIMEOUT) -> bytes:
    """The HTTP request itself (GET, or POST with data). Kept separate so the tests can replace it."""
    request = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout, context=_SSL) as response:
        return response.read()


def _get(url: str, *, data: bytes | None = None, headers: dict | None = None, timeout: float = TIMEOUT) -> bytes:
    body = _fetch(url, data=data, headers=headers, timeout=timeout)
    if body[:2] == b"\x1f\x8b":                           # INCOIS's app feeds are gzipped whatever was asked
        body = gzip.decompress(body)
    return body


def _last_copy(kind: str) -> dict | None:
    with _lock:
        hit = _memory.get(kind)
    if hit:
        return hit[1]
    path = CACHE_DIR / f"{kind}.json"
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def _rebuild(kind: str, build) -> dict:
    """Build, store and return a fresh copy. One build per kind at a time."""
    with _building[kind]:
        with _lock:
            hit = _memory.get(kind)
        if hit and time.time() - hit[0] < TTL[kind]:
            return hit[1]                                   # another request just built it
        now = time.time()
        result = build()
        result.update({"fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "fetched_epoch": now, "stale": False})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{kind}.json").write_text(json.dumps(result))
        with _lock:
            _memory[kind] = (now, result)
        return result


def _refresh_in_background(kind: str, build) -> None:
    if _building[kind].locked():
        return

    def run():
        try:
            _rebuild(kind, build)
        except Exception as exc:  # noqa: BLE001 -- the last copy keeps being served
            logger.warning("ocean connect %s: background refresh failed (%s)", kind, exc)
    threading.Thread(target=run, daemon=True, name=f"connect-{kind}").start()


def _cached(kind: str, build) -> dict:
    """Within the TTL, serve the copy in hand. Past it, serve that copy at once (marked as refreshing) and
    rebuild in the background; the INCOIS feeds can take minutes. With no copy at all, build now; if that
    fails there is nothing to serve."""
    copy = _last_copy(kind)
    if copy and time.time() - copy.get("fetched_epoch", 0) < TTL[kind]:
        with _lock:
            _memory.setdefault(kind, (copy["fetched_epoch"], copy))
        return copy
    if copy:
        _refresh_in_background(kind, build)
        age = time.time() - copy.get("fetched_epoch", 0)
        return {**copy, "refreshing": True, "stale": age > 3 * TTL[kind]}
    try:
        return _rebuild(kind, build)
    except Exception as exc:  # noqa: BLE001
        raise ConnectError(f"{kind} unavailable: {exc}") from exc


# --- news -------------------------------------------------------------------------

def _tags(title: str) -> list[str]:
    return [name for name, rx in TAGS if re.search(rx, title, re.I)]


def _build_news() -> dict:
    global _last_gdelt
    wait = GDELT_SPACING_S - (time.time() - _last_gdelt)
    if wait > 0:
        time.sleep(wait)
    _last_gdelt = time.time()
    params = {"query": GDELT_QUERY, "mode": "artlist", "maxrecords": 75, "format": "json",
              "timespan": "3d", "sort": "datedesc"}
    raw = _get(f"{GDELT_URL}?{urllib.parse.urlencode(params)}").decode("utf-8", "replace")
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise ConnectError(f"GDELT did not return JSON: {raw[:120]}") from exc
    seen, articles = set(), []
    for a in body.get("articles", []):
        if a.get("language") and a["language"] != "English":
            continue
        title = re.sub(r"\s+", " ", a.get("title", "")).strip()
        key = title.lower()
        if not title or key in seen:
            continue
        seen.add(key)
        when = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        articles.append({"title": title, "url": a.get("url"), "domain": a.get("domain"),
                         "country": a.get("sourcecountry") or None, "time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "image": a.get("socialimage") or None, "tags": _tags(title)})
    return {"articles": articles, "source": "GDELT DOC 2.0 API", "query": GDELT_QUERY, "window": "3 days",
            "note": "Articles are matched by phrase by GDELT's news index; they are not verified by this platform."}


def _build_news_rss(gdelt_error: str) -> dict:
    """The same search on Google News RSS, used only when GDELT refuses and nothing is cached."""
    params = {"q": f"{NEWS_TERMS} when:3d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    rss = ElementTree.fromstring(_get(f"{GNEWS_RSS}?{urllib.parse.urlencode(params)}"))
    seen, articles = set(), []
    for it in rss.iter("item"):
        source = it.find("source")
        outlet = (source.text or "").strip() if source is not None else ""
        title = re.sub(r"\s+", " ", it.findtext("title") or "").strip()
        if outlet and title.endswith(f" - {outlet}"):
            title = title[: -len(outlet) - 3]
        key = title.lower()
        if not title or key in seen:
            continue
        seen.add(key)
        try:
            when = datetime.strptime(it.findtext("pubDate") or "", "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        domain = urllib.parse.urlparse(source.get("url", "")).netloc.removeprefix("www.") if source is not None else None
        articles.append({"title": title, "url": it.findtext("link"), "domain": domain or outlet or None, "outlet": outlet or None,
                         "country": None, "time": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "image": None, "tags": _tags(title)})
    articles.sort(key=lambda a: a["time"], reverse=True)
    return {"articles": articles, "source": "Google News RSS (fallback)", "query": params["q"], "window": "3 days",
            "fallback_reason": f"GDELT unavailable: {gdelt_error}",
            "note": ("GDELT refused the request, so the same search was run on Google News RSS. Articles are matched by "
                     "the search engine; they are not verified by this platform.")}


def _build_news_any() -> dict:
    try:
        return _build_news()
    except Exception as exc:  # noqa: BLE001 -- a GDELT copy in hand is kept rather than replaced by the fallback
        copy = _last_copy("news")
        if copy and copy.get("source", "").startswith("GDELT"):
            raise
        logger.warning("ocean connect news: GDELT %s; using Google News RSS", exc)
        return _build_news_rss(str(exc))


def get_news() -> dict:
    news = _cached("news", _build_news_any)
    # Topics are tagged on the way out, so a change to TAGS applies to a cached copy too.
    return {**news, "articles": [{**a, "tags": _tags(a["title"])} for a in news["articles"]]}


# --- events ------------------------------------------------------------------------

def _tsunami_events() -> list[dict]:
    body = json.loads(_get(ITEWC_URL))
    rows = body.get("datasets", [])

    def detail(row):
        try:
            info = json.loads(_get(row["detail"]))[0]["event_info"][0]
            return info.get("evaluation"), info.get("bulletinTitle"), info.get("bulletinIssueTime")
        except Exception:  # noqa: BLE001 -- the list entry stands without its bulletin text
            return None, None, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        details = list(pool.map(detail, rows))
    events = []
    for row, (evaluation, title, issued) in zip(rows, details):
        origin = datetime.strptime(row["ORIGINTIME"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        threat = bool(evaluation) and not re.search(r"does not exist|no (tsunami )?threat|no threat", evaluation, re.I)
        events.append({
            "id": row["EVID"], "class": "TSUNAMI" if threat else "EARTHQUAKE",
            "colour": "#ef4444" if threat else "#94a3b8",
            "title": f"M{row['MAGNITUDE']} earthquake, {row['REGIONNAME']}",
            "text": evaluation or "Bulletin text unavailable.",
            "lat": float(row["LATITUDE"]), "lon": float(row["LONGITUDE"]),
            "place": row["REGIONNAME"], "time": origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "expires": None, "active": False,
            "sender": "INCOIS ITEWC", "severity": "threat" if threat else "no threat to India",
            "magnitude": float(row["MAGNITUDE"]), "depth_km": float(row["DEPTH"]),
            "link": row["detail"], "bulletin": title, "issued": issued,
        })
    return events


LEVELS = {"red": ("Warning", 3), "orange": ("Alert", 2), "yellow": ("Watch", 1), "green": ("No threat", 0)}
_WINDOW = re.compile(r"during\s+(\d\d:\d\d) hours on (\d\d-\d\d-\d{4}) to (\d\d:\d\d) hours on (\d\d-\d\d-\d{4})")


def _district_centres() -> dict[str, dict]:
    """Centre of each coastal district's alert polygon (from INCOIS), kept on disk for a month."""
    path = CACHE_DIR / "district_centres.json"
    if path.exists() and time.time() - path.stat().st_mtime < DISTRICTS_TTL_S:
        return json.loads(path.read_text())
    try:
        geo = json.loads(_get(DISTRICTS_URL, timeout=SLOW_TIMEOUT))
    except Exception:  # noqa: BLE001 -- an older copy is fine: district boundaries do not move
        if path.exists():
            return json.loads(path.read_text())
        raise
    centres = {}
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        ring = max((p[0] for p in polys), key=len)               # the main landmass of the district
        centres[f["properties"]["District"]] = {
            "state": f["properties"].get("STATE"),
            "lat": round(sum(pt[1] for pt in ring) / len(ring), 3),
            "lon": round(sum(pt[0] for pt in ring) / len(ring), 3)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(centres))
    return centres


def _incois_alerts() -> list[dict]:
    """INCOIS High Wave, Swell Surge and Ocean Current alerts, one event per bulletin and level, with every district."""
    rows = []
    hw = json.loads(_get(HWASSA_URL, timeout=SLOW_TIMEOUT))
    for date_key, json_key in (("LatestHWADate", "HWAJson"), ("LatestSSADate", "SSAJson")):
        if hw.get(date_key) not in (None, "None") and hw.get(json_key) not in (None, "None"):
            rows += json.loads(hw[json_key])
    try:
        cur = json.loads(_get(CURRENTS_URL, timeout=60))
        if cur.get("LatestCurrentsDate") not in (None, "None"):
            rows += json.loads(cur["CurrentsJson"])
    except Exception as exc:  # noqa: BLE001 -- wave alerts stand without the currents ones
        logger.warning("ocean connect: INCOIS currents alerts unavailable (%s)", exc)
    centres = _district_centres() if rows else {}
    now = datetime.now(timezone.utc)

    def utc(hhmm, dmy):
        return datetime.strptime(f"{dmy} {hhmm}", "%d-%m-%Y %H:%M").replace(tzinfo=IST).astimezone(timezone.utc)

    groups: dict[tuple, dict] = {}
    for r in rows:
        level, rank = LEVELS.get((r.get("Color") or "").lower(), (r.get("Color"), 0))
        if rank == 0:
            continue
        alert = (r.get("Alert") or "").upper()
        klass = _classify(alert)
        if not klass:
            continue
        m = _WINDOW.search(r.get("Message") or "")
        start, end = (utc(m.group(1), m.group(2)), utc(m.group(3), m.group(4))) if m else (None, None)
        c = centres.get(r["District"], {})
        area = {"district": r["District"].title(), "state": (r.get("STATE") or "").title(), "lat": c.get("lat"), "lon": c.get("lon"),
                "message": re.sub(r"\s+", " ", r.get("Message") or "").strip(),
                "from": start and start.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": end and end.strftime("%Y-%m-%dT%H:%M:%SZ")}
        key = (alert, level, r.get("Issue Date"))
        g = groups.setdefault(key, {"class": klass[0], "colour": klass[1], "alert": alert, "level": level, "rank": rank,
                                    "issued": r.get("Issue Date"), "areas": []})
        g["areas"].append(area)

    events = []
    for (alert, level, issued), g in groups.items():
        areas = sorted(g["areas"], key=lambda a: (a["state"], a["district"]))
        starts = [a["from"] for a in areas if a["from"]]
        ends = [a["to"] for a in areas if a["to"]]
        states = sorted({a["state"] for a in areas})
        placed = [a for a in areas if a["lat"] is not None]
        issued_iso = datetime.strptime(issued, "%d-%m-%Y").strftime("%Y-%m-%d") if issued else None
        events.append({
            "id": f"incois:{alert}:{level}:{issued}", "class": g["class"], "colour": g["colour"],
            "title": f"{alert.title()} — {len(areas)} coastal district{'s' if len(areas) != 1 else ''}",
            "text": areas[0]["message"] if len(areas) == 1 else
                    (f"{len(areas)} coastal districts in {', '.join(states)}, each with its own forecast window and range "
                     f"(listed per district). For example: {areas[0]['message']}"),
            # a single district is placed on the map; a multi-district bulletin is placed by its districts ("areas")
            "lat": placed[0]["lat"] if len(placed) == 1 else None,
            "lon": placed[0]["lon"] if len(placed) == 1 else None,
            "place": ", ".join(states), "time": min(starts) if starts else None, "expires": max(ends) if ends else None,
            "active": bool(ends) and max(ends) > now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sender": "INCOIS", "alert": alert, "severity": level.lower(), "level": level, "level_rank": g["rank"],
            "issued": issued_iso, "areas": areas, "link": "https://incois.gov.in/site/services/hwa.jsp",
        })
    return events


def _classify(text: str) -> tuple[str, str] | None:
    for name, colour, rx in EVENT_CLASSES:
        if re.search(rx, text, re.I):
            return name, colour
    return None


def _cap_event(item: dict) -> dict | None:
    ns = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
    root = ElementTree.fromstring(_get(item["link"]))
    infos = root.findall("cap:info", ns)
    info = next((i for i in infos if (i.findtext("cap:language", "", ns) or "").lower().startswith("en")), infos[0] if infos else None)
    if info is None:
        return None
    headline = (info.findtext("cap:headline", "", ns) or "").strip()
    event = (info.findtext("cap:event", "", ns) or "").strip()
    klass = _classify(f"{event} {headline}")
    if not klass:
        return None
    area = info.find("cap:area", ns)
    area_desc = (area.findtext("cap:areaDesc", "", ns) if area is not None else "") or ""
    lat = lon = None
    poly_url = next((p.findtext("cap:value", "", ns) for p in info.findall("cap:parameter", ns)
                     if p.findtext("cap:valueName", "", ns) == "Polygon URL"), None)
    if poly_url:
        try:
            text = ElementTree.fromstring(_get(poly_url)).findtext(".//polygon") or ""
            pts = [tuple(map(float, p.split(","))) for p in text.split() if "," in p]
            if pts:
                lat = round(sum(p[0] for p in pts) / len(pts), 3)
                lon = round(sum(p[1] for p in pts) / len(pts), 3)
        except Exception:  # noqa: BLE001 -- no polygon: listed without a map position
            pass
    sent = root.findtext("cap:sent", "", ns)
    expires = info.findtext("cap:expires", "", ns)

    def iso(s):
        try:
            return datetime.fromisoformat(s).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    exp = iso(expires) if expires else None
    return {
        "id": root.findtext("cap:identifier", "", ns), "class": klass[0], "colour": klass[1],
        "title": event or klass[0].title(), "text": headline, "lat": lat, "lon": lon, "place": area_desc,
        "time": iso(sent), "expires": exp,
        "active": bool(exp) and datetime.fromisoformat(exp.replace("Z", "+00:00")) > datetime.now(timezone.utc),
        "sender": root.findtext("cap:sender", "", ns) or item["author"],
        "severity": (info.findtext("cap:severity", "", ns) or "").lower() or None,
        "urgency": (info.findtext("cap:urgency", "", ns) or "").lower() or None,
        "link": item["link"],
    }


def _sachet_events() -> list[dict]:
    rss = ElementTree.fromstring(_get(SACHET_RSS))
    items = []
    for it in rss.iter("item"):
        author = (it.findtext("author") or "")
        if not MARINE_SENDERS.search(author):
            continue                                           # river gauges (CWC) and the like
        items.append({"link": it.findtext("link"), "author": author})
    with ThreadPoolExecutor(max_workers=4) as pool:
        events = list(pool.map(lambda i: _safe(_cap_event, i), items))
    return [e for e in events if e]


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ocean connect: skipping %s (%s)", arg.get("link") if isinstance(arg, dict) else arg, exc)
        return None


def _build_events() -> dict:
    errors = []
    sources = (("INCOIS coastal alerts", _incois_alerts), ("INCOIS ITEWC", _tsunami_events), ("NDMA SACHET", _sachet_events))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(name, pool.submit(fn)) for name, fn in sources]
    results = []
    for name, future in futures:
        try:
            results.append(future.result())
        except Exception as exc:  # noqa: BLE001 -- one source down leaves the others
            results.append([])
            errors.append(f"{name}: {exc}")
    coastal, tsunami, alerts = results
    if not coastal and not tsunami and not alerts and errors:
        raise ConnectError("; ".join(errors))
    events = sorted(coastal + tsunami + alerts, key=lambda e: e["time"] or "", reverse=True)
    return {"events": events, "errors": errors,
            "classes": [{"name": n, "colour": c} for n, c, _ in EVENT_CLASSES] + [{"name": "EARTHQUAKE", "colour": "#94a3b8"}],
            "levels": [{"name": v[0], "rank": v[1], "colour": k} for k, v in LEVELS.items() if v[1]],
            "sources": {"coastal": HWASSA_URL, "currents": CURRENTS_URL, "tsunami": ITEWC_URL, "alerts": SACHET_RSS},
            "note": ("Official bulletins and alerts: INCOIS High Wave, Swell Surge and Ocean Current alerts by coastal district "
                     "(INCOIS levels: Warning, Alert, Watch), INCOIS ITEWC earthquake/tsunami bulletins (past 90 days) and the coastal and marine "
                     "Common Alerting Protocol alerts of IMD, INCOIS and state authorities from NDMA SACHET. Positions of alerts are the "
                     "centre of the alert's area. For action, follow the issuing authority.")}


def get_events() -> dict:
    return _cached("events", _build_events)


# --- observations ------------------------------------------------------------------------

NETWORKS = {
    # key: (name, colour, what it is)
    "argo": ("Argo floats", "#e2e8f0", "Profiling floats that dive to 2,000 m and surface every ~10 days with temperature and salinity; "
                                          "the international Argo programme, to which INCOIS contributes India's floats. Floats that "
                                          "surfaced in the last 10 days, from this platform's Argo service."),
    "moored_met": ("Moored met buoys", "#38bdf8", "INCOIS/NIOT moored buoys reporting surface meteorology and ocean data."),
    "moored_omni": ("OMNI moored buoys (reporting)", "#0ea5e9", "NIOT's Ocean Moored buoy network for the North Indian Ocean: "
                                                                "surface met and subsurface temperature, salinity and currents."),
    "drifting": ("Drifting buoys", "#a3e635", "Surface drifters that follow the currents, reporting position, SST and air pressure."),
    "waverider": ("Wave-rider buoys", "#4ade80", "INCOIS coastal wave-rider buoys measuring wave height, period and direction."),
    "aws": ("Ship weather stations", "#f472b6", "Automatic weather stations on ships, reporting along their routes."),
    "rama": ("RAMA moored array", "#f59e0b", "Research Moored Array for African-Asian-Australian Monsoon Analysis and Prediction: "
                                            "deep-ocean moorings across the tropical Indian Ocean (NOAA PMEL with INCOIS and partners)."),
    "omni_sites": ("OMNI buoy sites", "#22d3ee", "The planned and deployed OMNI mooring sites, with INCOIS's reporting status."),
    "tide_gauge": ("Tide gauges", "#c084fc", "Coastal sea-level stations used for tides and tsunami detection."),
    "hf_radar": ("HF radar sites", "#fb7185", "Shore-based high-frequency radars mapping surface currents out to ~200 km."),
}
ARGO_BOX = (30.0, -40.0, 120.0, 30.0)                   # lon_min, lat_min, lon_max, lat_max
ARGO_DAYS = 10


def _oon_platforms(start, end) -> list[dict]:
    body = json.dumps({"startDate": start.isoformat(), "endDate": end.isoformat(),
                       "moored": True, "aws": True, "drifting": True, "waverider": True}).encode()
    latest: dict[str, dict] = {}
    for r in json.loads(_get(OON_URL, data=body, headers={"Content-Type": "application/json"})):
        pid = r.get("buoy_id") or r.get("ship_name")
        key = f"{r['category']}:{pid}"
        if key not in latest or r["time"] > latest[key]["time"]:
            latest[key] = {"network": r["category"], "id": pid, "lat": r["lat"], "lon": r["lon"],
                           "time": r["time"].replace(" ", "T") + "Z", "reporting": True}
    return list(latest.values())


def _wfs_platforms(network: str, ws: str, layer: str) -> list[dict]:
    out = []
    for f in json.loads(_get(WFS.format(ws=ws, layer=layer)))["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        pid = p.get("ID") or p.get("Station Na") or p.get("S.No.")
        status = p.get("Reporting")
        out.append({"network": network, "id": str(pid), "lat": lat, "lon": lon, "time": None,
                    "reporting": None if status is None else status.lower() == "reporting", "agency": p.get("Agency")})
    return out


def _hf_radar_platforms() -> list[dict]:
    return [{"network": "hf_radar", "id": s["site_location"], "lat": s["latitude"], "lon": s["longitude"], "time": None,
             "reporting": None, "agency": s.get("state")} for s in json.loads(_get(HF_RADAR_URL))]


def _argo_platforms() -> list[dict]:
    from app.services import argo
    floats = argo.fetch_floats_in_bbox(*ARGO_BOX, days=ARGO_DAYS)
    return [{"network": "argo", "id": str(f["wmo"]), "lat": round(float(f["lat"]), 3), "lon": round(float(f["lon"]), 3),
             "time": str(f["last_seen"]).replace(" ", "T").split("+")[0].rstrip("Z") + "Z", "reporting": True,
             "cycle": f.get("cycle")} for f in floats]


def _build_observations() -> dict:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=3)
    sources = [("Ocean Observation Network", lambda: _oon_platforms(start, end)),
               ("RAMA", lambda: _wfs_platforms("rama", "JointPortal", "JointPortal:Ramabuoys")),
               ("OMNI sites", lambda: _wfs_platforms("omni_sites", "JointPortal", "JointPortal:Omni_Buoy")),
               ("Tide gauges", lambda: _wfs_platforms("tide_gauge", "Insitu_TideGauges_Tsunami", "Insitu_TideGauges_Tsunami:Tideguages57")),
               ("HF radar", _hf_radar_platforms),
               ("Argo", _argo_platforms)]
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = [(name, pool.submit(fn)) for name, fn in sources]
    platforms: dict[str, dict] = {}
    errors = []
    for name, future in futures:
        try:
            for p in future.result():
                platforms[f"{p['network']}:{p['id']}"] = p
        except Exception as exc:  # noqa: BLE001 -- one network down leaves the others
            errors.append(f"{name}: {exc}")
    if not platforms:
        raise ConnectError("; ".join(errors) or "no observing platforms returned")

    items = sorted(platforms.values(), key=lambda p: (p["network"], p["id"]))
    summary = []
    for key, (name, colour, about) in NETWORKS.items():
        group = [p for p in items if p["network"] == key]
        if not group:
            continue
        times = [p["time"] for p in group if p["time"]]
        summary.append({"network": key, "name": name, "colour": colour, "about": about, "count": len(group),
                        "reporting": sum(1 for p in group if p["reporting"]) if any(p["reporting"] is not None for p in group) else None,
                        "latest": max(times) if times else None,
                        "window": (f"surfaced in the last {ARGO_DAYS} days" if key == "argo" else
                                   "reported in the last 3 days" if times else "network sites")})
    return {"platforms": items, "networks": summary, "errors": errors, "window": f"{start} to {end}",
            "note": ("INCOIS Ocean Observation Network: platforms that reported in the last three days (latest position), the "
                     "RAMA/OMNI arrays, tide gauges and HF radar sites from INCOIS's map services, and Argo floats that surfaced in "
                     f"the last {ARGO_DAYS} days. Positions only; the measurements are served by INCOIS and the Argo data centres.")}


def get_observations() -> dict:
    return _cached("observations", _build_observations)


# --- hazards (the platform's own 72-hour outlooks) ---------------------------------------

def _peak(outlook: dict, k: int, pick: str) -> dict | None:
    """Where the chosen extreme of frame k sits: (lat, lon, value)."""
    g, frame = outlook["grid"], outlook["frames"][k]
    best = None
    for n, v in enumerate(frame):
        if v is None:
            continue
        if best is None or (v > best[1] if pick == "max" else v < best[1]):
            best = (n, v)
    if best is None:
        return None
    j, i = divmod(best[0], g["nx"])
    return {"lat": round(g["lat0"] + j * g["dlat"], 2), "lon": round(g["lon0"] + i * g["dlon"], 2), "value": best[1]}


HAZARD_READINGS = {
    # hazard: (summary field for the headline, its label, units, extreme to locate, how the frame's value reads)
    "cyclones": ("max_wind_kt", "strongest forecast wind", "kt", "max", "kt wind"),
    "heatwaves": ("max_multiple", "hottest water, as a multiple of the heatwave threshold", "×", "max", "× threshold"),
    "low_oxygen": ("shallowest_m", "shallowest hypoxic water", "m", "min", "m to hypoxia"),
    "algal_blooms": ("max_mg_m3", "highest forecast chlorophyll", "mg/m³", "max", "mg/m³ chlorophyll"),
}


def get_hazard_highlights() -> dict:
    """One line per hazard from the Ocean Hazards tab's 72-hour outlooks: now, the peak over 72 h, and where."""
    from app.services import hazards
    items, errors = [], []
    for key, (field, label, units, pick, reads) in HAZARD_READINGS.items():
        try:
            o = hazards.get_outlook(key)
        except Exception as exc:  # noqa: BLE001 -- a hazard that cannot be built is listed as unavailable
            errors.append(f"{key}: {exc}")
            continue
        s = o["summaries"]
        values = [x.get(field) for x in s]
        known = [(k, v) for k, v in enumerate(values) if v is not None]
        if not known:
            continue
        k_peak, v_peak = (max if pick == "max" else min)(known, key=lambda t: t[1])
        where = _peak(o, k_peak, pick)
        items.append({
            "hazard": key, "title": o["title"], "label": label, "units": units, "reads": reads,
            "now": values[0], "peak": v_peak, "peak_time": o["times"][k_peak], "peak_lead_h": o["leads_h"][k_peak],
            "area_label": o["summary_label"], "area_now_km2": s[0]["area_km2"], "area_peak_km2": max(x["area_km2"] for x in s),
            "where": where, "issued": o["issued"], "source": o["sources"][0], "time_kind": o["time_kind"],
        })
    return {"hazards": items, "errors": errors,
            "note": ("From this platform's Ocean Hazards tab: forecast models (INCOIS wave/wind, Copernicus Marine), read by the rules "
                     "stated there. They are outlooks, not warnings; for warnings, follow INCOIS and IMD.")}
