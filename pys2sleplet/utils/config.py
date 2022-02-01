from pathlib import Path

from box import Box

_file_location = Path(__file__).resolve()
_settings_file = _file_location.parent / "config" / "settings.toml"

settings = Box.from_toml(filename=_settings_file)
