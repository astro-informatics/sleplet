import logging
import os

import pooch

import sleplet._vars

logger = logging.getLogger(__name__)


ZENODO_DATA_DOI = "10.5281/zenodo.7767698"
POOCH = pooch.create(
    path=pooch.os_cache("sleplet"),
    base_url=f"doi:{ZENODO_DATA_DOI}/",
    registry=None,
)
POOCH.load_registry_from_doi()


def find_on_pooch_then_local(filename: str) -> os.PathLike | None:
    """Find a file on POOCH first and if not look in data folder."""
    if filename in POOCH.registry:
        logger.info(f"Found {filename} at https://doi.org/{ZENODO_DATA_DOI}")
        return POOCH.fetch(filename, progressbar=True)
    if (sleplet._vars.DATA_PATH / filename).exists():
        logger.info(f"Found {filename} at {sleplet._vars.DATA_PATH / filename}")
        return sleplet._vars.DATA_PATH / filename
    logger.info(f"No {filename} found, calculating...")
    return None
