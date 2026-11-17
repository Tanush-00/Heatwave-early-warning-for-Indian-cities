"""Smoke tests: the whole scoring step on made-up data, the README update, and the app."""
import pandas as pd
import pytest

import config
import evaluate
from conftest import ROOT


def test_evaluate_on_made_up_data(backtest, rows, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    t = evaluate.evaluate(backtest, rows=rows, write_readme=False, log=lambda *_: None)
    for name in ["mae_heat_season.csv", "lgbm_vs_best_baseline.csv", "heatwave_brier.csv",
                 "heatwave_warnings.csv", "interval_coverage.csv", "reliability.csv", "summary.md",
                 "feature_importance.csv"]:
        assert (tmp_path / "results" / name).exists(), name
    for name in ["mae_by_lead.png", "reliability.png", "city_mae.png", "delhi_2024.png"]:
        assert (tmp_path / "images" / name).stat().st_size > 10_000, name
    assert list(t["mae_heat"].index) == config.LEADS


def test_md_table():
    df = pd.DataFrame({"a": [1, 2], "b": [0.123, 4.5]})
    assert evaluate.md_table(df) == "| a | b |\n|---|---|\n| 1 | 0.12 |\n| 2 | 4.50 |"


def test_update_readme_replaces_only_the_results(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{evaluate.START}\nold\n{evaluate.END}\noutro\n", encoding="utf-8")
    monkeypatch.setattr(evaluate, "README", readme)
    assert evaluate.update_readme("new $1 \\n numbers")           # regex-special text stays literal
    assert readme.read_text(encoding="utf-8") == \
        f"intro\n{evaluate.START}\nnew $1 \\n numbers\n{evaluate.END}\noutro\n"
    readme.write_text("no markers", encoding="utf-8")
    assert not evaluate.update_readme("x")


@pytest.mark.skipif(not config.PREDICTIONS_CSV.exists(), reason="run `python run_all.py` first")
def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    at.selectbox[0].set_value("jaipur").run()                    # city explorer: another city
    at.toggle[0].set_value(False).run()                          # ...the whole year
    assert not at.exception, at.exception
