from pathlib import Path

import numpy as np
from dynaconf import Dynaconf

from pys2sleplet.utils.region import Region

_file_location = Path(__file__).resolve()
_settings_file = _file_location.parents[1] / "config" / "settings.toml"

settings = Dynaconf(settings_files=[_settings_file])
default_region = Region(
    gap=settings.POLAR_GAP,
    mask_name=settings.SLEPIAN_MASK,
    order=settings.ORDER,
    phi_max=np.deg2rad(settings.PHI_MAX),
    phi_min=np.deg2rad(settings.PHI_MIN),
    theta_max=np.deg2rad(settings.THETA_MAX),
    theta_min=np.deg2rad(settings.THETA_MIN),
)
