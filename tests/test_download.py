"""Download code, tested with a fake internet (no network is used)."""
import hashlib
import json
from datetime import date

import pytest

import config
import download

CITY = {"id": "testcity", "name": "Test", "station": "00000", "lat": 28.5, "lon": 77.1}


def test_forecast_url_asks_for_what_we_need():
    url = download.forecast_url(CITY, "2024-01-01", "2024-12-31")
    for d in range(1, config.MAX_PREVIOUS_DAY + 1):
        assert f"temperature_2m_previous_day{d}" in url
    assert "models=gfs_seamless" in url and "timezone=Asia%2FKolkata" in url
    assert "latitude=28.5" in url and "start_date=2024-01-01" in url
    assert "key" not in url.lower()            # free source: no API key anywhere


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CITIES", [CITY])
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "STATION_DIR", tmp_path / "stations")
    monkeypatch.setattr(config, "FORECAST_DIR", tmp_path / "forecasts")
    monkeypatch.setattr(config, "MANIFEST_JSON", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "FORECAST_START", "2025-11-01")
    monkeypatch.setattr(config, "API_PAUSE_SECONDS", 0)
    calls = []

    def fake_fetch(url, tries=4):
        calls.append(url)
        if "bulk.meteostat" in url:
            return b"station bytes"
        return json.dumps({"hourly": {"time": ["2025-11-01T00:00"], "temperature_2m_previous_day1": [20.0]}}).encode()

    monkeypatch.setattr(download, "fetch", fake_fetch)
    return tmp_path, calls


def test_download_forecasts_one_file_per_year_and_skips_finished_years(fake_world):
    tmp, calls = fake_world
    download.download_forecasts(date(2026, 1, 15), log=lambda *_: None)
    assert sorted(p.name for p in (tmp / "forecasts").iterdir()) == ["testcity_2025.json", "testcity_2026.json"]
    assert "start_date=2025-11-01" in calls[0] and "end_date=2025-12-31" in calls[0]
    assert "start_date=2026-01-01" in calls[1] and "end_date=2026-01-15" in calls[1]
    calls.clear()
    download.download_forecasts(date(2026, 1, 20), log=lambda *_: None)
    assert len(calls) == 1 and "start_date=2026-01-01" in calls[0]     # only the current year again


def test_download_stations_skips_existing(fake_world):
    tmp, calls = fake_world
    download.download_stations(log=lambda *_: None)
    download.download_stations(log=lambda *_: None)
    assert len(calls) == 1
    assert (tmp / "stations" / "00000.csv.gz").read_bytes() == b"station bytes"


def test_answer_without_data_is_an_error(fake_world, monkeypatch):
    monkeypatch.setattr(download, "fetch", lambda url, tries=4: b'{"error": true, "reason": "bad"}')
    with pytest.raises(RuntimeError, match="without data"):
        download.download_forecasts(date(2026, 1, 15), log=lambda *_: None)


def test_manifest_fingerprints(fake_world):
    tmp, _ = fake_world
    download.download_stations(log=lambda *_: None)
    download.download_forecasts(date(2025, 12, 1), log=lambda *_: None)
    m = download.write_manifest()
    assert set(m["files"]) == {"stations/00000.csv.gz", "forecasts/testcity_2025.json"}
    assert m["files"]["stations/00000.csv.gz"] == hashlib.sha256(b"station bytes").hexdigest()
    assert json.loads((tmp / "manifest.json").read_text())["files"] == m["files"]
