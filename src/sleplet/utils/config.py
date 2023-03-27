from __future__ import annotations

from pathlib import Path

import tomli

_settings_path = Path(__file__).resolve().parents[1] / "config"

with open(_settings_path / "settings.toml", "rb") as f:
    settings = tomli.load(f)
