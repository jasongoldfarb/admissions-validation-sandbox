from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def get_output_dir(output_dir: Optional[PathLike] = None) -> Path:
    """Return the requested output directory or the repository's outputs folder."""
    if output_dir is not None:
        return Path(output_dir)
    return Path(__file__).resolve().parents[1] / "outputs"
