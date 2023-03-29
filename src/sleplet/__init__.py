"""
API of the `SLEPLET` package. See [PyPI](https://pypi.org/project/sleplet) for
the rest of the documentation.

[![PyPI](https://badge.fury.io/py/sleplet.svg)](https://pypi.org/project/sleplet)
"""

import logging
import os

from sleplet._version import __version__

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] --- %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

NCPU = int(os.getenv("NCPU", "4"))
PHI_MAX = int(os.getenv("PHI_MAX", "360"))
PHI_MIN = int(os.getenv("PHI_MIN", "0"))
POLAR_GAP = os.getenv("POLAR_GAP", "False").lower() == "true"
SLEPIAN_MASK = os.getenv("SLEPIAN_MASK", "south_america")
THETA_MAX = int(os.getenv("THETA_MAX", "180"))
THETA_MIN = int(os.getenv("THETA_MIN", "0"))

logger.info(
    "Environment variables set as "
    f"NCPU={NCPU}, "
    f"POLAR_GAP={POLAR_GAP}, "
    f"THETA_MAX={THETA_MAX}, "
    f"THETA_MIN={THETA_MIN}, "
    f"PHI_MAX={PHI_MAX}, "
    f"PHI_MIN={PHI_MIN}, "
    f"SLEPIAN_MASK={SLEPIAN_MASK}."
)
