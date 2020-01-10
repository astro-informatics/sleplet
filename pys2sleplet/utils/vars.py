from pathlib import Path

from .misc import load_config

__parent_dir = Path(__file__).resolve().parents[1]

ENVS = load_config(__parent_dir / "config.toml")
SLEPIAN = load_config(__parent_dir / "slepian" / "slepian.toml")

PHI_MIN_DEFAULT = 0
PHI_MAX_DEFAULT = 180
THETA_MIN_DEFAULT = 0
THETA_MAX_DEFAULT = 360
