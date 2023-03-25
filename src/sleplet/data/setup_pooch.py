import os
from pathlib import Path

import pooch

from sleplet.utils.vars import ZENODO_DATA_DOI

_data_path = Path(__file__).resolve().parent

POOCH = pooch.create(
    path=pooch.os_cache("sleplet"),
    # Use the figshare DOI
    base_url=f"doi:{ZENODO_DATA_DOI}/",
    registry=None,
)
POOCH.load_registry_from_doi()


def find_on_pooch_then_local(filename: str) -> os.PathLike | None:
    """
    find a file on POOCH first and if not look in data folder
    """
    if filename in POOCH.registry:
        return POOCH.fetch(filename)
    elif (_data_path / filename).exists():
        return _data_path / filename
    else:
        return None
