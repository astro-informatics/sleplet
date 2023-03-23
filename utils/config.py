from pathlib import Path

import tomli

_file_location = Path(__file__).resolve()
_settings_file = _file_location.parents[1] / "config" / "settings.toml"

with open(_settings_file, "rb") as f:
    settings = tomli.load(f)
