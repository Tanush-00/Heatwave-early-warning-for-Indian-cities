"""The whole pipeline in one command.

    python run_all.py                  # download -> dataset -> backtest -> evaluate
    python run_all.py --skip-download  # reuse the files already in data/raw/

Then: streamlit run app.py
"""
import argparse
import sys
import time

import config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true", help="use the raw files already downloaded")
    args = parser.parse_args(argv)
    t0 = time.time()

    if not args.skip_download:
        import download
        print("[1/4] Downloading (free sources, no key) ...", flush=True)
        download.main()
    elif not any(config.STATION_DIR.glob("*.csv.gz")):
        print("No raw data in data/raw/. Run without --skip-download first.")
        return 1

    import dataset
    print("[2/4] Cleaning and building the daily table ...", flush=True)
    if dataset.main() != 0:
        return 1

    from backtest import run_backtest
    from features import make_rows
    print("\n[3/4] Walk-forward backtest (monthly retraining) ...", flush=True)
    rows = make_rows(dataset.load_daily())
    fc = run_backtest(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_csv(config.PREDICTIONS_CSV, index=False, float_format="%.4f")
    print(f"  {len(fc):,} forecasts -> {config.PREDICTIONS_CSV.relative_to(config.ROOT)}", flush=True)

    from evaluate import evaluate
    print("\n[4/4] Scoring ...", flush=True)
    evaluate(fc, rows=rows, log=lambda *_: None)
    print(f"  results/, images/ and the README results section updated.\n\nDone in {time.time() - t0:.0f} s. "
          "Next: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
