import os
from pathlib import Path

from dynaconf import LazySettings

config = LazySettings(
    SETTINGS_FILE_FOR_DYNACONF=Path(__file__).resolve().parents[1]
    / "config"
    / "settings.toml",
    ENV_FOR_DYNACONF=os.getenv("ENV_FOR_DYNACONF", "default"),
)
