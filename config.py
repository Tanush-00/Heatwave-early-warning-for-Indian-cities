"""All settings in one place. Change things here, not inside the other files."""
from pathlib import Path

# --- Folders -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STATION_DIR = RAW_DIR / "stations"        # one .csv.gz per weather station (Meteostat)
FORECAST_DIR = RAW_DIR / "forecasts"      # one .json per city per year (Open-Meteo)
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "images"

MANIFEST_JSON = RAW_DIR / "manifest.json"
DAILY_CSV = PROCESSED_DIR / "daily.csv"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"

# --- Data sources (both free, no key) -------------------------------------------
# Observed temperature at IMD weather stations (airport stations report every hour or
# half hour), via Meteostat bulk data. We use the HOURLY file and build the daily maximum
# ourselves: Meteostat's ready-made daily file had some months shifted by one day.
METEOSTAT_URL = "https://bulk.meteostat.net/v2/hourly/{station}.csv.gz"
# Old weather-model forecasts: Open-Meteo "Previous Runs" API. For every hour it keeps
# what the model said 1, 2, ... days earlier. GFS is the only model archived from 2021.
OPEN_METEO_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "gfs_seamless"
FORECAST_START = "2021-04-01"            # first day the archive has GFS forecasts
API_PAUSE_SECONDS = 1.0                   # be polite to free services

# --- Cities (inland plains, where IMD's 40 degC heatwave rule applies) ------------
# station = WMO / Meteostat id. Coordinates are the station's, so the model forecast
# is for the same point as the thermometer.
CITIES = [
    {"id": "delhi",      "name": "Delhi (Palam airport)", "station": "42181", "lat": 28.5667, "lon": 77.1167},
    {"id": "jaipur",     "name": "Jaipur (Sanganer)",    "station": "42348", "lat": 26.8167, "lon": 75.8000},
    {"id": "jodhpur",    "name": "Jodhpur",              "station": "42339", "lat": 26.3000, "lon": 73.0167},
    {"id": "ganganagar", "name": "Sri Ganganagar",       "station": "42123", "lat": 29.9167, "lon": 73.9167},
    {"id": "lucknow",    "name": "Lucknow (Amausi)",     "station": "42369", "lat": 26.7500, "lon": 80.8833},
    {"id": "varanasi",   "name": "Varanasi (Babatpur)",  "station": "42479", "lat": 25.4500, "lon": 82.8667},
    {"id": "patna",      "name": "Patna",                "station": "42492", "lat": 25.6000, "lon": 85.1000},
    {"id": "gwalior",    "name": "Gwalior",              "station": "42361", "lat": 26.2333, "lon": 78.2500},
    {"id": "jhansi",     "name": "Jhansi",               "station": "42463", "lat": 25.4500, "lon": 78.5833},
    {"id": "kota",       "name": "Kota",                 "station": "42452", "lat": 25.1500, "lon": 75.8500},
    {"id": "bhopal",     "name": "Bhopal (Bairagarh)",   "station": "42667", "lat": 23.2833, "lon": 77.3500},
    {"id": "nagpur",     "name": "Nagpur (Sonegaon)",    "station": "42867", "lat": 21.1000, "lon": 79.0500},
    {"id": "akola",      "name": "Akola",                "station": "42933", "lat": 20.7000, "lon": 77.0333},
    {"id": "chandrapur", "name": "Chandrapur",           "station": "43029", "lat": 19.9667, "lon": 79.3000},
    {"id": "hyderabad",  "name": "Hyderabad (Airport)",  "station": "43128", "lat": 17.4500, "lon": 78.4667},
]
CITY_IDS = [c["id"] for c in CITIES]

# --- Forecast set-up -------------------------------------------------------------
# A forecast is "made" on the evening of day t (the day's max temperature is known by
# then) for day t + lead. Leads 1, 3 and 5 days ahead.
LEADS = [1, 3, 5]
MAX_PREVIOUS_DAY = max(LEADS) + 1         # we also need the run one day older (for "revision")
DAYTIME_HOURS = range(10, 19)             # 10:00-18:00 IST: the day's maximum happens here

# --- Cleaning and city selection ----------------------------------------------------
# "tmax" in this project = the highest hourly reading between 10:00 and 18:00 IST: the
# same window as the model forecast. It is a little below the official max-thermometer value.
TEMP_RANGE = (-10.0, 52.0)                # India's record is 51.0 degC (Phalodi, 2016)
MIN_READINGS_DAYTIME = 3                  # a day needs >= 3 readings between 10:00 and 18:00
MIN_READINGS_DAY = 6                      # ...and >= 6 in the whole day for its minimum
SPIKE_DEGREES = 10.0                      # one day 10 degC away from BOTH neighbours = typing mistake
NORMAL_YEARS = (1991, 2020)               # climate normal, like IMD's reference period
MIN_NORMAL_YEARS = 10                     # a city needs >= 10 full years of data in that period
                                          # (smoothed over 31 days: >= 300 values per calendar day)
MIN_COVERAGE = 0.8                        # ...and a max temperature on >= 80% of days since FORECAST_START

# --- IMD heatwave rule (plains), applied to one station on one day --------------------
# heatwave if  (Tmax >= 40 and Tmax - normal >= 4.5)  or  Tmax >= 45
HEAT_MIN_TMAX = 40.0
HEAT_DEPARTURE = 4.5
HEAT_ABSOLUTE = 45.0
HEAT_SEASON_MONTHS = [3, 4, 5, 6]         # March-June: when heatwave scores are measured

# --- Backtest (walk-forward) -----------------------------------------------------------
TEST_START = "2023-01-01"                 # 2021-04 .. 2022-12 is only for the first training
RETRAIN_EVERY_DAYS = 30                   # retrain monthly, only on targets already observed
BIAS_WINDOW_DAYS = 14                     # "recent bias" of the weather model (classic correction)
INTERVAL_LOW, INTERVAL_HIGH = 0.10, 0.90  # 80% interval
CONFORMAL_WINDOW_DAYS = 365               # past errors used to calibrate the interval

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)

# --- Statistics --------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_DAYS = 7                  # resample whole weeks (errors on nearby days are linked)
SEED = 42
