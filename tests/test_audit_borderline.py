import pandas as pd
import pytest

from src import audit_borderline


@pytest.mark.parametrize(
    "prob,expected",
    [
        (0.0, "1_clear_reject (<0.20)"),
        (0.199, "1_clear_reject (<0.20)"),
        (0.20, "2_borderline (0.20-0.65)"),
        (0.50, "2_borderline (0.20-0.65)"),
        (0.65, "2_borderline (0.20-0.65)"),
        (0.651, "3_clear_admit (>0.65)"),
        (1.0, "3_clear_admit (>0.65)"),
    ],
)
def test_assign_zone_default_boundaries(prob, expected):
    """Verify inclusive borderline bounds [0.20, 0.65] and outer zones."""
    assert audit_borderline.assign_zone(prob) == expected


def test_assign_zone_custom_thresholds():
    """Custom low/high cutoffs shift zone assignment as expected."""
    assert audit_borderline.assign_zone(0.10, low=0.15, high=0.80) == "1_clear_reject (<0.20)"
    assert audit_borderline.assign_zone(0.75, low=0.15, high=0.80) == "2_borderline (0.20-0.65)"
    assert audit_borderline.assign_zone(0.85, low=0.15, high=0.80) == "3_clear_admit (>0.65)"


def test_main_raises_when_baseline_missing(tmp_path, monkeypatch):
    """Fail fast with a clear FileNotFoundError if Session 1 output is missing."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="baseline_applicants.csv"):
        audit_borderline.main()


def test_main_writes_audited_csv_with_derived_columns(tmp_path, monkeypatch):
    """main() partitions all rows into zones and computes ap_utilization and wrongful_reject."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    sample_df = pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4],
            "cohort": ["traditional", "traditional", "non_traditional", "non_traditional"],
            "gpa": [3.9, 3.2, 3.6, 3.4],
            "test_score": [1450, 1050, 1200, 980],
            "ap_courses_taken": [8, 2, 3, 0],
            "school_ap_cap": [10, 10, 4, 0],
            "outside_hours_wk": [2.0, 5.0, 18.0, 15.0],
            "upward_trajectory": [0, 0, 1, 1],
            "true_qualified": [1, 0, 1, 0],
            "pred_prob": [0.85, 0.35, 0.12, 0.05],
            "pred_admit": [1, 0, 0, 0],
        }
    )
    sample_df.to_csv(out_dir / "baseline_applicants.csv", index=False)

    audit_borderline.main()

    audited_path = out_dir / "audited_applicants.csv"
    assert audited_path.exists()

    audited = pd.read_csv(audited_path)
    assert len(audited) == len(sample_df)
    assert {"zone", "ap_utilization", "wrongful_reject"}.issubset(audited.columns)
    assert audited["zone"].isna().sum() == 0
    assert audited["wrongful_reject"].tolist() == [0, 0, 1, 0]
    assert audited["ap_utilization"].tolist() == pytest.approx([0.8, 0.2, 0.75, 0.0])
