from pathlib import Path

from dynaconf import Dynaconf

_file_location = Path(__file__).resolve()
_settings_file = _file_location.parents[1] / "config" / "settings.toml"

settings = Dynaconf(settings_files=[_settings_file])
