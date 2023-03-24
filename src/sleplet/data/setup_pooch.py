import os
from pathlib import Path

import pooch

from sleplet.utils.vars import ZENODO_DATA_DOI

POOCH = pooch.create(
    path=pooch.os_cache("sleplet"),
    # Use the figshare DOI
    base_url=f"doi:{ZENODO_DATA_DOI}/",
    registry=None,
)
POOCH.load_registry_from_doi()


def find_on_pooch_then_local(data_path: Path, filename: str) -> os.PathLike:
    """
    find a file on POOCH first and if not look in data folder
    """
    return POOCH.fetch(filename) if filename in POOCH.registry else data_path / filename
