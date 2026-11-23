import pandas as pd
import pytest

from src import baseline_repro


def test_generate_applicants_reproducible():
    """Same seed produces bit-for-bit identical applicant DataFrames."""
    df1 = baseline_repro.generate_applicants(n_samples=500, seed=42)
    df2 = baseline_repro.generate_applicants(n_samples=500, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_applicants_schema_and_bounds():
    """Generated dataset has expected columns, no nulls, and valid feature bounds."""
    df = baseline_repro.generate_applicants(n_samples=2_000, seed=42)

    expected_cols = {
        "applicant_id",
        "cohort",
        "is_nontraditional",
        "gpa",
        "test_score",
        "ap_courses_taken",
        "school_ap_cap",
        "leadership_index",
        "outside_hours_wk",
        "upward_trajectory",
        "true_qualified",
        "historical_admit",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 2_000
    assert df.isna().sum().sum() == 0

    assert df["gpa"].between(2.0, 4.0).all()
    assert df["test_score"].between(400, 1600).all()
    assert (df["test_score"] % 10 == 0).all()
    assert (df["ap_courses_taken"] >= 0).all()
    assert (df["ap_courses_taken"] <= df["school_ap_cap"]).all()
    assert df["leadership_index"].between(1.0, 10.0).all()
    assert set(df["cohort"].unique()) == {"traditional", "non_traditional"}


def test_generate_applicants_matches_empirical_priors():
    """Large sample aligns with NCES share, College Board SAT priors, and 20% admit bar."""
    df = baseline_repro.generate_applicants(n_samples=10_000, seed=42)

    assert df["is_nontraditional"].mean() == pytest.approx(
        baseline_repro.FRACTION_OF_PROTECTED, abs=0.02
    )
    assert df["test_score"].mean() == pytest.approx(baseline_repro.SAT_MEAN, abs=15.0)
    assert df["test_score"].std() == pytest.approx(baseline_repro.SAT_STD, abs=15.0)
    assert df["true_qualified"].mean() == pytest.approx(
        baseline_repro.TIER1_ADMIT_RATE, abs=0.005
    )
    assert df["historical_admit"].mean() == pytest.approx(
        baseline_repro.TIER1_ADMIT_RATE, abs=0.005
    )


def test_summarize_metrics_exact_values():
    """Verify accuracy, FNR, TPR, and missed count on a known confusion matrix."""
    df = pd.DataFrame(
        {
            "true_qualified": [1, 1, 1, 0, 0],
            "pred_admit": [1, 0, 0, 0, 1],
        }
    )
    metrics = baseline_repro.summarize_metrics(df)

    assert metrics["n"] == 5
    assert metrics["accuracy"] == pytest.approx(2 / 5)
    assert metrics["fnr_wrongful_reject"] == pytest.approx(2 / 3)
    assert metrics["tpr_equal_opportunity"] == pytest.approx(1 / 3)
    assert metrics["missed_qualified_count"] == 2


def test_summarize_metrics_zero_positives_edge_case():
    """No division-by-zero when a slice contains zero true positives."""
    df = pd.DataFrame({"true_qualified": [0, 0, 0], "pred_admit": [0, 1, 0]})
    metrics = baseline_repro.summarize_metrics(df)

    assert metrics["fnr_wrongful_reject"] == 0.0
    assert metrics["tpr_equal_opportunity"] == 0.0
    assert metrics["missed_qualified_count"] == 0


def test_main_writes_holdout_csv(tmp_path, monkeypatch):
    """main() writes outputs/baseline_applicants.csv in an isolated temp directory."""
    monkeypatch.chdir(tmp_path)
    baseline_repro.main()

    out_file = tmp_path / "outputs" / "baseline_applicants.csv"
    assert out_file.exists()

    holdout = pd.read_csv(out_file)
    assert len(holdout) == 3_000
    assert {"pred_prob", "pred_admit"}.issubset(holdout.columns)
    assert holdout["pred_prob"].between(0.0, 1.0).all()
