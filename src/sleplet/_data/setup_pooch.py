import os
from pathlib import Path

import pooch

import sleplet

_data_path = Path(__file__).resolve().parent

ZENODO_DATA_DOI = "10.5281/zenodo.7767698"
POOCH = pooch.create(
    path=pooch.os_cache("sleplet"),
    base_url=f"doi:{ZENODO_DATA_DOI}/",
    registry=None,
)
POOCH.load_registry_from_doi()


def find_on_pooch_then_local(filename: str) -> os.PathLike | None:
    """find a file on POOCH first and if not look in data folder."""
    if filename in POOCH.registry:
        sleplet.logger.info(f"Found {filename} at https://doi.org/{ZENODO_DATA_DOI}")
        return POOCH.fetch(filename, progressbar=True)
    if (_data_path / filename).exists():
        sleplet.logger.info(f"Found {filename} at {_data_path / filename}")
        return _data_path / filename
    sleplet.logger.info(f"No {filename} found, calculating...")
    return None
