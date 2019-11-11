from pathlib import Path

import toml


def load_config():
    filename = Path(__file__).resolve().parent / "config.toml"
    config = toml.load(filename)
    return config


ENVS = load_config()
