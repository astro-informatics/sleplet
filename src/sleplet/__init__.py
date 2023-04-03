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

_logger = logging.getLogger(__name__)

_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] --- %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)
_logger.setLevel("INFO")
_logger.propagate = False

__all__ = [
    "__version__",
    "harmonic_methods",
    "noise",
    "plot_methods",
    "plotting",
    "slepian_methods",
    "wavelet_methods",
]
