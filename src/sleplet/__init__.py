"""
API of the `SLEPLET` package. See [PyPI](https://pypi.org/project/sleplet) for
the rest of the documentation.
"""

import logging

from . import (
    harmonic_methods,
    noise,
    plot_methods,
    plotting,
    slepian_methods,
    wavelet_methods,
)
from ._version import __version__

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

__all__ = [
    "__version__",
    "harmonic_methods",
    "noise",
    "plot_methods",
    "plotting",
    "slepian_methods",
    "wavelet_methods",
]
