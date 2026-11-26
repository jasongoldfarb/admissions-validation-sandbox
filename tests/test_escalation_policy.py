import numpy as np
import pandas as pd
import pytest

from src import escalation_policy


def _sample_audited_df(n: int = 100) -> pd.DataFrame:
    """Create a deterministic, balanced sample DataFrame for policy tests."""
    rng = np.random.default_rng(7)
    cohorts = np.where(np.arange(n) % 2 == 0, "traditional", "non_traditional")
    true_qual = (rng.uniform(size=n) < 0.30).astype(int)
    pred_prob = np.clip(rng.uniform(0.0, 1.0, size=n), 0.0, 1.0)
    pred_admit = (pred_prob >= 0.50).astype(int)

    return pd.DataFrame(
        {
            "applicant_id": np.arange(1, n + 1),
            "cohort": cohorts,
            "school_ap_cap": np.where(cohorts == "non_traditional", 4, 12),
            "outside_hours_wk": np.where(cohorts == "non_traditional", 15.0, 2.0),
            "true_qualified": true_qual,
            "pred_prob": pred_prob,
            "pred_admit": pred_admit,
        }
    )


def test_evaluate_policy_reproducible():
    """Same seed produces identical policy evaluation metrics."""
    df = _sample_audited_df()
    mask = df["pred_prob"].between(0.20, 0.65)

    res1 = escalation_policy.evaluate_policy(df, mask, human_review_accuracy=0.92, seed=42)
    res2 = escalation_policy.evaluate_policy(df, mask, human_review_accuracy=0.92, seed=42)
    assert res1 == res2


def test_zero_escalation_matches_baseline():
    """Escalating 0 rows leaves predictions untouched regardless of human accuracy."""
    df = _sample_audited_df()
    empty_mask = pd.Series(False, index=df.index)

    res_low = escalation_policy.evaluate_policy(
        df, empty_mask, human_review_accuracy=0.50, seed=42, capacity_fraction=1.0
    )
    res_high = escalation_policy.evaluate_policy(
        df, empty_mask, human_review_accuracy=1.00, seed=42, capacity_fraction=1.0
    )

    assert res_low == res_high
    assert res_low["escalated_count"] == 0
    assert res_low["escalated_pct"] == 0.0
    assert res_low["overall_accuracy"] == pytest.approx(
        (df["true_qualified"] == df["pred_admit"]).mean()
    )


def test_full_escalation_with_perfect_human_reviewer():
    """Perfect review makes only qualified applicants eligible within the seat cap."""
    df = _sample_audited_df()
    full_mask = pd.Series(True, index=df.index)

    res = escalation_policy.evaluate_policy(
        df, full_mask, human_review_accuracy=1.0, seed=42, capacity_fraction=1.0
    )

    assert res["escalated_count"] == len(df)
    assert res["escalated_pct"] == pytest.approx(1.0)
    assert res["overall_accuracy"] == pytest.approx(1.0)
    assert res["trad_fnr"] == pytest.approx(0.0)
    assert res["nontrad_fnr"] == pytest.approx(0.0)


def test_higher_human_accuracy_improves_overall_accuracy():
    """Higher human accuracy improves selection accuracy when all seats can be filled."""
    df = _sample_audited_df(n=500)
    full_mask = pd.Series(True, index=df.index)

    low_acc = escalation_policy.evaluate_policy(
        df, full_mask, human_review_accuracy=0.60, seed=42, capacity_fraction=1.0
    )
    high_acc = escalation_policy.evaluate_policy(
        df, full_mask, human_review_accuracy=0.95, seed=42, capacity_fraction=1.0
    )

    assert high_acc["overall_accuracy"] > low_acc["overall_accuracy"]


def test_evaluate_policy_respects_capacity_and_reports_unfilled_seats():
    df = _sample_audited_df(n=20)
    no_escalation = pd.Series(False, index=df.index)

    result = escalation_policy.evaluate_policy(
        df, no_escalation, capacity_fraction=0.10
    )

    assert result["capacity_seats"] == 2
    assert result["admitted_count"] <= result["capacity_seats"]
    assert result["unfilled_seats"] == result["capacity_seats"] - result["admitted_count"]
    assert result["eligible_count"] == int(df["pred_admit"].sum())


def test_main_raises_when_audited_missing(tmp_path, monkeypatch):
    """Fail fast with FileNotFoundError if Session 2 output is missing."""
    with pytest.raises(FileNotFoundError, match="audited_applicants.csv"):
        escalation_policy.main(tmp_path / "outputs")


def test_main_writes_capacity_tradeoff_csv(tmp_path):
    """main() writes policy metrics with a shared seat-cap target."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    _sample_audited_df(n=200).to_csv(out_dir / "audited_applicants.csv", index=False)

    escalation_policy.main(out_dir, capacity_fraction=0.10)

    out_file = out_dir / "escalation_tradeoff.csv"
    assert out_file.exists()

    tradeoff = pd.read_csv(out_file)
    assert len(tradeoff) == 8
    assert {
        "policy",
        "escalated_pct",
        "escalated_count",
        "capacity_seats",
        "model_shortlisted_count",
        "eligible_count",
        "admitted_count",
        "unfilled_seats",
        "capacity_fill_pct",
        "overall_accuracy",
        "trad_fnr",
        "nontrad_fnr",
    }.issubset(tradeoff.columns)

    for col in ["escalated_pct", "overall_accuracy", "trad_fnr", "nontrad_fnr"]:
        assert tradeoff[col].between(0.0, 1.0).all()
    assert (tradeoff["admitted_count"] <= tradeoff["capacity_seats"]).all()
