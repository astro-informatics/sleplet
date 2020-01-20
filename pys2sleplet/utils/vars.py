from pathlib import Path

from .misc import load_config

ENVS = load_config(Path(__file__).resolve().parents[1] / "config.toml")

PHI_MIN_DEFAULT = 0
PHI_MAX_DEFAULT = 180
THETA_MIN_DEFAULT = 0
THETA_MAX_DEFAULT = 360
