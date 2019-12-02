from pathlib import Path

import toml


def load_config(filename):
    return toml.load(filename)


parent_dir = Path(__file__).resolve().parent

ENVS = load_config(parent_dir / "config.toml")
SLEPIAN = load_config(parent_dir / "slepian" / "slepian.toml")
