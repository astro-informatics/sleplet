from dataclasses import KW_ONLY, field
from pathlib import Path
from typing import Optional

import cmocean
import numpy as np
import plotly.offline as py
import pyssht as ssht
from plotly.graph_objs import Figure, Surface
from plotly.graph_objs.surface import Lighting
from pydantic.dataclasses import dataclass

from sleplet.utils.config import settings
from sleplet.utils.logger import logger
from sleplet.utils.plot_methods import (
    boost_field,
    calc_plot_resolution,
    convert_colourscale,
    create_plot_type,
    normalise_function,
    set_outside_region_to_minimum,
)
from sleplet.utils.plotly_methods import (
    create_camera,
    create_colour_bar,
    create_layout,
    create_tick_mark,
)
from sleplet.utils.region import Region
from sleplet.utils.vars import SAMPLING_SCHEME, SPHERE_UNSEEN

MW_POLE_LENGTH = 2

_file_location = Path(__file__).resolve()
_fig_path = _file_location.parents[1] / "figures"


@dataclass
class Plot:
    f: np.ndarray
    L: int
    filename: str
    _: KW_ONLY
    amplitude: Optional[float] = None
    annotations: list[dict] = field(default_factory=list)
    normalise: bool = True
    plot_type: str = "real"
    reality: bool = False
    region: Optional[Region] = None
    spin: int = 0
    upsample: bool = True

    def __post_init__(self) -> None:
        self.resolution = calc_plot_resolution(self.L) if self.upsample else self.L
        if self.upsample:
            self.filename += f"_res{self.resolution}"
        self.filename += f"_{self.plot_type}"
        if self.normalise:
            self.filename += "_norm"

    def execute(self) -> None:
        """
        creates basic plotly plot rather than matplotlib
        """
        f = self._prepare_field(self.f)

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self._setup_plot(
            f, self.resolution, method=SAMPLING_SCHEME
        )

        if isinstance(self.region, Region):
            # make plot area clearer
            f_plot = set_outside_region_to_minimum(f_plot, self.resolution, self.region)

        # appropriate zoom in on north pole
        camera = create_camera(-0.1, -0.1, 10, 7.88)

        # pick largest tick max value
        tick_mark = create_tick_mark(vmin, vmax, amplitude=self.amplitude)

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                cmax=1 if self.normalise else tick_mark,
                cmid=0.5 if self.normalise else 0,
                cmin=0 if self.normalise else -tick_mark,
                colorbar=create_colour_bar(tick_mark, self.normalise),
                colorscale=convert_colourscale(cmocean.cm.ice),
                lighting=Lighting(ambient=1),
                reversescale=True,
            )
        ]

        layout = create_layout(camera, annotations=self.annotations)

        fig = Figure(data=data, layout=layout)

        # create html and open if auto_open is true
        html_filename = str(_fig_path / "html" / f"{self.filename}.html")

        py.plot(fig, filename=html_filename, auto_open=settings.AUTO_OPEN)

        # if save_fig is true then create png and pdf in their directories
        if settings.SAVE_FIG:
            for file_type in {"png", "pdf"}:
                logger.info(f"saving {file_type}")
                filename = str(_fig_path / file_type / f"{self.filename}.{file_type}")
                fig.write_image(filename, engine="kaleido")

    @staticmethod
    def _setup_plot(
        f: np.ndarray,
        resolution: int,
        method: str = "MW",
        close: bool = True,
        parametric: bool = False,
        parametric_scaling: Optional[list[float]] = None,
        color_range: Optional[list[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        function which creates the data for the matplotlib/plotly plot
        """
        if parametric_scaling is None:
            parametric_scaling = [0.0, 0.5]
        if method == "MW_pole":
            if len(f) == MW_POLE_LENGTH:
                f, _ = f
            else:
                f, _, _ = f

        thetas, phis = ssht.sample_positions(resolution, Grid=True, Method=method)

        if thetas.size != f.size:
            raise AttributeError("Bandlimit L deos not match that of f")

        f_plot = f.copy()

        f_max = f_plot.max()
        f_min = f_plot.min()

        if not isinstance(color_range, list):
            vmin = f_min
            vmax = f_max
        else:
            vmin = color_range[0]
            vmax = color_range[1]
            f_plot[f_plot < color_range[0]] = color_range[0]
            f_plot[f_plot > color_range[1]] = color_range[1]
            f_plot[f_plot == SPHERE_UNSEEN] = np.nan

        # Compute position scaling for parametric plot.
        f_normalised = (
            (f_plot - vmin / (vmax - vmin)) * parametric_scaling[1]
            + parametric_scaling[0]
            if parametric
            else np.zeros(f_plot.shape)
        )

        # Close plot.
        if close:
            _, n_phi = ssht.sample_shape(resolution, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, 0], axis=1)
            if parametric:
                f_normalised = np.insert(
                    f_normalised, n_phi, f_normalised[:, 0], axis=1
                )
            thetas = np.insert(thetas, n_phi, thetas[:, 0], axis=1)
            phis = np.insert(phis, n_phi, phis[:, 0], axis=1)

        # Compute location of vertices.
        x, y, z = (
            ssht.spherical_to_cart(f_normalised, thetas, phis)
            if parametric
            else ssht.s2_to_cart(thetas, phis)
        )

        return x, y, z, f_plot, vmin, vmax

    def _prepare_field(self, f: np.ndarray) -> np.ndarray:
        """
        boosts, forces plot type and then scales the field before plotting
        """
        boosted_field = boost_field(
            f, self.L, self.resolution, self.reality, self.spin, self.upsample
        )
        field_space = create_plot_type(boosted_field, self.plot_type)
        return normalise_function(field_space, self.normalise)
