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


def test_run_simulation_reproducible_with_separate_random_streams():
    """Each regime is repeatable with fixed seeds."""
    for regime in ("baseline", "context_aware_hitl"):
        first = longitudinal_sim.run_simulation(
            n_samples=200, n_cycles=5, regime=regime, seed=42
        )
        second = longitudinal_sim.run_simulation(
            n_samples=200, n_cycles=5, regime=regime, seed=42
        )
        pd.testing.assert_frame_equal(first, second)


def test_environmental_random_streams_match_across_regimes(monkeypatch):
    """Review draws do not shift shared feature-generation or shock streams."""
    feature_states = []
    shock_states = []
    original_features = longitudinal_sim.compute_cycle_features
    original_updates = longitudinal_sim.apply_quality_updates

    def capture_features(state_df, rng):
        feature_states.append(repr(rng.bit_generator.state))
        return original_features(state_df, rng)

    def capture_updates(*args, **kwargs):
        shock_states.append(repr(kwargs["rng"].bit_generator.state))
        return original_updates(*args, **kwargs)

    monkeypatch.setattr(longitudinal_sim, "compute_cycle_features", capture_features)
    monkeypatch.setattr(longitudinal_sim, "apply_quality_updates", capture_updates)

    longitudinal_sim.run_simulation(n_samples=100, n_cycles=3, regime="baseline", seed=42)
    baseline_feature_states = feature_states.copy()
    baseline_shock_states = shock_states.copy()

    feature_states.clear()
    shock_states.clear()
    longitudinal_sim.run_simulation(
        n_samples=100, n_cycles=3, regime="context_aware_hitl", seed=42
    )

    assert feature_states == baseline_feature_states
    assert shock_states == baseline_shock_states


def test_cycle_qualification_uses_contextual_readiness(monkeypatch):
    """The longitudinal cycle passes the same context signals as the baseline scorer."""
    state = longitudinal_sim.initialize_ecosystem(n_samples=20, seed=42)
    observed = {}
    original = longitudinal_sim.compute_true_readiness_score

    def capture_readiness(z_acad, z_nonacad, ap_utilization, outside_hours, upward, rng):
        observed["ap_utilization"] = ap_utilization.copy()
        observed["outside_hours"] = outside_hours.copy()
        observed["upward"] = upward.copy()
        return original(z_acad, z_nonacad, ap_utilization, outside_hours, upward, rng)

    monkeypatch.setattr(longitudinal_sim, "compute_true_readiness_score", capture_readiness)
    result = longitudinal_sim.compute_cycle_features(state, np.random.default_rng(7))

    assert observed["ap_utilization"] == pytest.approx(
        result["ap_courses_taken"].to_numpy()
        / np.maximum(result["school_ap_cap"].to_numpy(), 1)
    )
    assert observed["outside_hours"] == pytest.approx(state["outside_hours_wk"].to_numpy())
    assert observed["upward"] == pytest.approx(state["upward_trajectory"].to_numpy())


def test_main_writes_longitudinal_csv(tmp_path, monkeypatch):
    """main() writes outputs/longitudinal_dynamics.csv with 100 rows (50 per regime)."""
    longitudinal_sim.main(tmp_path / "outputs")

    out_file = tmp_path / "outputs" / "longitudinal_dynamics.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 100
    assert set(df["regime"].unique()) == {"baseline", "context_aware_hitl"}
