from pathlib import Path

from dynaconf import Dynaconf

_file_location = Path(__file__).resolve()

settings = Dynaconf(
    settings_files=[_file_location.parents[1] / "config" / "settings.toml"]
)
