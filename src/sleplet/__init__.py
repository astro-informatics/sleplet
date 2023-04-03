"""
API of the `SLEPLET` package. See [PyPI](https://pypi.org/project/sleplet) for
the rest of the documentation.
"""

import logging

from . import (
    functions,
    harmonic_methods,
    meshes,
    noise,
    plot_methods,
    plotting,
    slepian,
    slepian_methods,
    wavelet_methods,
)
from ._version import __version__  # noqa: F401

__all__ = [
    "functions",
    "harmonic_methods",
    "meshes",
    "noise",
    "plot_methods",
    "plotting",
    "slepian_methods",
    "slepian",
    "wavelet_methods",
]

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
