"""
API of the `SLEPLET` package. See [PyPI](https://pypi.org/project/sleplet) for
the rest of the documentation.
"""

import logging

from ._version import __version__  # noqa: F401

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] --- %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
