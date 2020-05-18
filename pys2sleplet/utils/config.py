import os
from pathlib import Path

from dynaconf import LazySettings

_file_location = Path(__file__).resolve().parents[1]

config = LazySettings(
    SETTINGS_FILE_FOR_DYNACONF=_file_location / "config" / "settings.toml",
    ENV_FOR_DYNACONF=os.getenv("ENV_FOR_DYNACONF", "default"),
)
