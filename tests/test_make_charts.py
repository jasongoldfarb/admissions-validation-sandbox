import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # Headless backend for CI and local pytest runs

from src import make_charts

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal valid audited and tradeoff DataFrames for chart rendering tests."""
    audited_df = pd.DataFrame(
        {
            "true_qualified": [1, 1, 0, 0, 1, 0],
            "pred_admit": [1, 0, 0, 0, 0, 1],
        }
    )
    tradeoff_df = pd.DataFrame(
        {
            "policy": [
                "No Escalation (Baseline)",
                "Context-Aware Band (AP Cap <=6 or Work >=10h: [0.04,0.65] / Else: [0.20,0.65])",
            ],
            "escalated_pct": [0.0, 0.064],
            "overall_accuracy": [0.825, 0.855],
            "trad_fnr": [0.120, 0.084],
            "nontrad_fnr": [0.685, 0.500],
        }
    )
    return audited_df, tradeoff_df


def test_make_chart1_writes_valid_png(tmp_path):
    """make_chart1 writes a non-empty PNG file with a valid PNG header."""
    audited_df, tradeoff_df = _sample_inputs()
    out_path = tmp_path / "chart1.png"

    make_charts.make_chart1(audited_df, tradeoff_df, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.read_bytes().startswith(PNG_HEADER)


def test_make_chart2_writes_valid_png(tmp_path):
    """make_chart2 writes a non-empty PNG file with a valid PNG header."""
    audited_df, tradeoff_df = _sample_inputs()
    out_path = tmp_path / "chart2.png"

    make_charts.make_chart2(audited_df, tradeoff_df, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.read_bytes().startswith(PNG_HEADER)


def test_make_chart2_fallback_when_accuracy_col_missing(tmp_path):
    """make_chart2 falls back cleanly to audited_df accuracy if tradeoff_df omits it."""
    audited_df, tradeoff_df = _sample_inputs()
    tradeoff_no_acc = tradeoff_df.drop(columns=["overall_accuracy"])
    out_path = tmp_path / "chart2_fallback.png"

    make_charts.make_chart2(audited_df, tradeoff_no_acc, out_path)

    assert out_path.exists()
    assert out_path.read_bytes().startswith(PNG_HEADER)


def test_main_raises_when_inputs_missing(tmp_path, monkeypatch):
    """Fail fast with FileNotFoundError if Session 2/3 CSVs are missing."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing Session 2/3 CSV outputs"):
        make_charts.main()


def test_main_generates_both_charts(tmp_path, monkeypatch):
    """main() reads CSVs from outputs/ and writes both PNG charts."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    audited_df, tradeoff_df = _sample_inputs()
    audited_df.to_csv(out_dir / "audited_applicants.csv", index=False)
    tradeoff_df.to_csv(out_dir / "escalation_tradeoff.csv", index=False)

    make_charts.main()

    c1 = out_dir / "chart1_aggregate_vs_subgroup.png"
    c2 = out_dir / "chart2_escalation_tradeoff.png"
    assert c1.exists() and c1.read_bytes().startswith(PNG_HEADER)
    assert c2.exists() and c2.read_bytes().startswith(PNG_HEADER)
