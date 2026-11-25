import subprocess
import sys
from pathlib import Path


def test_pipeline_runs_from_outside_repository(tmp_path):
    """The script entry point creates all outputs from a different working directory."""
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "src" / "run_pipeline.py"
    output_dir = tmp_path / "generated"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Saved charts to:" in result.stdout
    expected_outputs = {
        "baseline_applicants.csv",
        "audited_applicants.csv",
        "escalation_tradeoff.csv",
        "longitudinal_dynamics.csv",
        "chart1_aggregate_vs_subgroup.png",
        "chart2_escalation_tradeoff.png",
        "chart3_longitudinal_dynamics.png",
    }
    assert expected_outputs.issubset({path.name for path in output_dir.iterdir()})
