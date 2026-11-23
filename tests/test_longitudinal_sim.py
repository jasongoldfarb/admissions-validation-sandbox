import numpy as np
import pandas as pd
import pytest

from src import longitudinal_sim


def test_initialize_ecosystem_reproducible_and_bounded():
    """Same seed produces identical initial populations with valid quality bounds."""
    df1 = longitudinal_sim.initialize_ecosystem(n_samples=200, seed=42)
    df2 = longitudinal_sim.initialize_ecosystem(n_samples=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)
    assert df1["quality_acad"].between(0.0, 1.0).all()
    assert df1["quality_nonacad"].between(0.0, 1.0).all()


def test_apply_quality_updates_exact_math():
    """TP gets +0.05, FP gets -0.05, rejected stays flat (with decay=0, shock=0)."""
    q = np.array([0.50, 0.50, 0.50, 0.98, 0.02])
    y_true = np.array([1, 0, 1, 1, 0])
    admit = np.array([1, 1, 0, 1, 1])

    updated = longitudinal_sim.apply_quality_updates(
        q, y_true, admit, delta_q_pos=0.05, delta_q_neg=-0.05, decay=0.0, shock_std=0.0
    )
    assert updated == pytest.approx([0.55, 0.45, 0.50, 1.00, 0.00])


def test_run_simulation_schema_and_regimes():
    """run_simulation returns expected cycle rows and valid metric bounds."""
    base = longitudinal_sim.run_simulation(n_samples=200, n_cycles=5, regime="baseline", seed=42)
    hitl = longitudinal_sim.run_simulation(
        n_samples=200, n_cycles=5, regime="context_aware_hitl", seed=42
    )

    assert len(base) == 5 and len(hitl) == 5
    assert base["cycle"].tolist() == [1, 2, 3, 4, 5]
    assert (base["escalated_pct"] == 0.0).all()
    assert hitl["escalated_pct"].mean() > 0.0
    assert base["overall_accuracy"].between(0.0, 1.0).all()
    assert hitl["nontrad_fnr"].between(0.0, 1.0).all()


def test_run_simulation_invalid_regime_raises():
    """Unknown regime raises ValueError."""
    with pytest.raises(ValueError, match="Unknown regime"):
        longitudinal_sim.run_simulation(n_samples=100, n_cycles=2, regime="invalid")


def test_main_writes_longitudinal_csv(tmp_path, monkeypatch):
    """main() writes outputs/longitudinal_dynamics.csv with 100 rows (50 per regime)."""
    monkeypatch.chdir(tmp_path)
    longitudinal_sim.main()

    out_file = tmp_path / "outputs" / "longitudinal_dynamics.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 100
    assert set(df["regime"].unique()) == {"baseline", "context_aware_hitl"}
