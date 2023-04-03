"""
API of the `SLEPLET` package. See [PyPI](https://pypi.org/project/sleplet) for
the rest of the documentation.
"""

import logging

from ._version import __version__  # noqa: F401

logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] --- %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel("INFO")
logger.propagate = False
