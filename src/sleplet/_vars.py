import os

import numpy as np

from sleplet import logger

NCPU = int(os.getenv("NCPU", "4"))
PHI_0 = np.pi
PHI_MAX_DEFAULT = 2 * np.pi
PHI_MAX = int(os.getenv("PHI_MAX", "360"))
PHI_MIN_DEFAULT = 0.0
PHI_MIN = int(os.getenv("PHI_MIN", "0"))
POLAR_GAP = os.getenv("POLAR_GAP", "False").lower() == "true"
RANDOM_SEED = 30
SAMPLING_SCHEME = "MWSS"
SLEPIAN_MASK = os.getenv("SLEPIAN_MASK", "south_america")
SPHERE_UNSEEN = -1.56e30
THETA_0 = 0.0
THETA_MAX_DEFAULT = np.pi
THETA_MAX = int(os.getenv("THETA_MAX", "180"))
THETA_MIN_DEFAULT = 0
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
