import logging
import os

import platformdirs
import pooch

_logger = logging.getLogger(__name__)


_ZENODO_DATA_DOI = "10.5281/zenodo.7767698"
_POOCH = pooch.create(
    path=pooch.os_cache("sleplet"),
    base_url=f"doi:{_ZENODO_DATA_DOI}/",
    registry=None,
)
_POOCH.load_registry_from_doi()


def find_on_pooch_then_local(filename: str) -> os.PathLike[str] | None:
    """Find a file on POOCH first and if not look in data folder."""
    if filename in _POOCH.registry:
        msg = f"Found {filename} at https://doi.org/{_ZENODO_DATA_DOI}"
        _logger.info(msg)
        return _POOCH.fetch(filename, progressbar=True)
    if (platformdirs.user_data_path() / filename).exists():
        msg = f"Found {filename} at {platformdirs.user_data_path() / filename}"
        _logger.info(msg)
        return platformdirs.user_data_path() / filename
    msg = f"No {filename} found, calculating..."
    _logger.info(msg)
    return None
