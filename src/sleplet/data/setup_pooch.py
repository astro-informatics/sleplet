import pooch

from sleplet.utils.vars import ZENODO_DATA_DOI

POOCH = pooch.create(
    path=pooch.os_cache("sleplet"),
    # Use the figshare DOI
    base_url=f"doi:{ZENODO_DATA_DOI}/",
    registry=None,
)
POOCH.load_registry_from_doi()
