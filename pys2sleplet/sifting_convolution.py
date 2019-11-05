import os

import numpy as np

from pys2sleplet.sphere import Sphere
from pys2sleplet.utils.plot_methods import calc_resolution
from pys2sleplet.utils.string_methods import missing_key


class SiftingConvolution:
    def __init__(
        self,
        flm: np.ndarray,
        flm_name: str,
        config: dict,
        glm: np.ndarray = None,
        glm_name: str = None,
    ) -> None:
        self.annotations = (
            config["annotations"]
            if "annotations" in config and config["annotation"]
            else []
        )
        self.auto_open = config["auto_open"]
        self.flm_name = flm_name
        self.flm = flm
        self.glm = glm
        if self.glm is not None:
            self.glm_name = glm_name
        self.L = config["L"]
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        self.sphere = Sphere(auto_open=config["auto_open"], save_fig=config["save_fig"])
        self.reality = config["reality"]
        self.save_fig = config["save_fig"]
        self.resolution = calc_resolution(config["L"])
        # missing_key(config, "routine", None)
        # missing_key(config, "type", None)
