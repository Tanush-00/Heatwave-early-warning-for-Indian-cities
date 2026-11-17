"""Step 1: download the raw data (both sources are free and need no key).

    python download.py

1. Observed daily max temperature for each station (Meteostat bulk file, one per station).
2. Old weather-model forecasts for each city (Open-Meteo Previous Runs API, one file per
   city per year): what the GFS model said 1..6 days before each hour.

Files that are already there are not downloaded again, except the current year's forecasts
(they keep growing). A SHA-256 fingerprint of every file goes into data/raw/manifest.json.
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

import config

USER_AGENT = "heatwave-early-warning (student project)"


def fetch(url: str, tries: int = 4) -> bytes:
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == tries:
                raise RuntimeError(f"Download failed after {tries} tries: {url}: {e}") from e
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def forecast_url(city: dict, start: str, end: str) -> str:
    hourly = ",".join(f"temperature_2m_previous_day{d}" for d in range(1, config.MAX_PREVIOUS_DAY + 1))
    params = {"latitude": city["lat"], "longitude": city["lon"], "hourly": hourly,
              "models": config.FORECAST_MODEL, "timezone": "Asia/Kolkata",
              "start_date": start, "end_date": end}
    return f"{config.OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"


def download_stations(log=print) -> None:
    config.STATION_DIR.mkdir(parents=True, exist_ok=True)
    for c in config.CITIES:
        path = config.STATION_DIR / f"{c['station']}.csv.gz"
        if path.exists():
            continue
        path.write_bytes(fetch(config.METEOSTAT_URL.format(station=c["station"])))
        log(f"  station {c['station']} ({c['name']}): {path.stat().st_size / 1e3:.0f} kB")
        time.sleep(config.API_PAUSE_SECONDS)


def download_forecasts(today: date, log=print) -> None:
    config.FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    first = date.fromisoformat(config.FORECAST_START)
    for c in config.CITIES:
        for year in range(first.year, today.year + 1):
            path = config.FORECAST_DIR / f"{c['id']}_{year}.json"
            if path.exists() and year < today.year:
                continue
            start = max(first, date(year, 1, 1))
            end = min(today, date(year, 12, 31))
            body = json.loads(fetch(forecast_url(c, start.isoformat(), end.isoformat())))
            if "hourly" not in body:
                raise RuntimeError(f"Open-Meteo answered without data for {c['id']} {year}: {str(body)[:200]}")
            path.write_text(json.dumps(body["hourly"]), encoding="utf-8")
            log(f"  forecasts {c['id']} {year}: {len(body['hourly']['time']):,} hours")
            time.sleep(config.API_PAUSE_SECONDS)


def write_manifest() -> dict:
    files = sorted(list(config.STATION_DIR.glob("*.csv.gz")) + list(config.FORECAST_DIR.glob("*.json")))
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"stations": config.METEOSTAT_URL, "forecasts": config.OPEN_METEO_URL,
                    "forecast_model": config.FORECAST_MODEL},
        "files": {str(p.relative_to(config.RAW_DIR)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in files},
    }
    config.MANIFEST_JSON.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    print("Downloading station observations (Meteostat) ...", flush=True)
    download_stations()
    print("Downloading old weather-model forecasts (Open-Meteo) ...", flush=True)
    download_forecasts(date.today())
    m = write_manifest()
    print(f"Done: {len(m['files'])} files, fingerprints in {config.MANIFEST_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
