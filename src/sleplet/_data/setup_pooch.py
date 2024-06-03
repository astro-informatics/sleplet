import logging
import os
import typing

import platformdirs
import pooch

_logger = logging.getLogger(__name__)


_POOCH = None
_POOCH_RETRY = 3
_ZENODO_DATA_DOI = "10.5281/zenodo.7767698"


def _lazy_load_registry(
    func: typing.Callable[..., os.PathLike[str] | None],
) -> typing.Callable[..., os.PathLike[str] | None]:
    """
    Lazily loads POOCH registry before executing a function.

    Args:
        func: Function to be decorated

    Returns:
        Decorated function
    """

    def wrapper(
        *args: typing.Any,  # noqa: ANN401
        **kwargs: typing.Any,  # noqa: ANN401
    ) -> os.PathLike[str] | None:
        """
        Load the POOCH registry if not already loaded and execute the function.

        Args:
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function.

        Returns:
            Result of the decorated function
        """
        global _POOCH  # noqa: PLW0603
        if _POOCH is None:
            _POOCH = pooch.create(
                path=pooch.os_cache("sleplet"),
                base_url=f"doi:{_ZENODO_DATA_DOI}/",
                registry=None,
                retry_if_failed=_POOCH_RETRY,
            )
            _POOCH.load_registry_from_doi()
        return func(*args, **kwargs)

    return wrapper


@_lazy_load_registry
def find_on_pooch_then_local(filename: str) -> os.PathLike[str] | None:
    """
    Find a file on POOCH first and if not look in data folder.

    Args:
        filename: Filename to find

    Returns:
        The sought after file or nothing if not found
    """
    if filename in _POOCH.registry:  # type: ignore[union-attr]
        msg = f"Found {filename} at https://doi.org/{_ZENODO_DATA_DOI}"
        _logger.info(msg)
        return _POOCH.fetch(filename, progressbar=True)  # type: ignore[union-attr]

    if (platformdirs.user_data_path() / filename).exists():
        msg = f"Found {filename} at {platformdirs.user_data_path() / filename}"
        _logger.info(msg)
        return platformdirs.user_data_path() / filename

    msg = f"No {filename} found, calculating..."
    _logger.info(msg)
    return None
