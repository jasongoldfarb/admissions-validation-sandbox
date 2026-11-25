"""Run the complete training sandbox and generate its CSV and chart outputs."""

import argparse
from pathlib import Path
from typing import Optional

if __package__:
    from . import audit_borderline, baseline_repro, escalation_policy, longitudinal_sim, make_charts
    from .paths import get_output_dir
else:
    import audit_borderline
    import baseline_repro
    import escalation_policy
    import longitudinal_sim
    import make_charts
    from paths import get_output_dir


def run_pipeline(output_dir: Optional[Path] = None) -> Path:
    """Run every simulation stage in order and return the generated output directory."""
    output_dir = get_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_repro.main(output_dir)
    audit_borderline.main(output_dir)
    escalation_policy.main(output_dir)
    longitudinal_sim.main(output_dir)
    make_charts.main(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for generated CSVs and charts (default: repository outputs/)",
    )
    args = parser.parse_args()
    run_pipeline(args.output_dir)


if __name__ == "__main__":
    main()
