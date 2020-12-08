from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cmocean
import numpy as np
import plotly.io as pio
import plotly.offline as py
import pyssht as ssht
from plotly.graph_objs import Figure, Layout, Surface
from plotly.graph_objs.layout import Margin, Scene
from plotly.graph_objs.layout.scene import Camera, XAxis, YAxis, ZAxis
from plotly.graph_objs.layout.scene.camera import Eye
from plotly.graph_objs.surface import ColorBar, Lighting
from plotly.graph_objs.surface.colorbar import Tickfont

from pys2sleplet.utils.config import settings
from pys2sleplet.utils.harmonic_methods import invert_flm_boosted
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import calc_plot_resolution, convert_colourscale
from pys2sleplet.utils.vars import SAMPLING_SCHEME, ZOOM_DEFAULT

_file_location = Path(__file__).resolve()
_fig_path = _file_location.parents[1] / "figures"


@dataclass
class Plot:
    f: np.ndarray = field(repr=False)
    L: int
    filename: str
    amplitude: Optional[float] = field(default=None, repr=False)
    plot_type: str = field(default="real", repr=False)
    annotations: List[Dict] = field(default_factory=list, repr=False)
    reality: bool = field(default=False, repr=False)
    spin: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.resolution = calc_plot_resolution(self.L) if settings.UPSAMPLE else self.L
        if settings.UPSAMPLE:
            self.filename += f"_res{self.resolution}"
        self.filename += f"_{self.plot_type}"

    def execute(self) -> None:
        """
        creates basic plotly plot rather than matplotlib
        """
        f = self._prepare_field(self.f)

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self._setup_plot(
            f, self.resolution, method=SAMPLING_SCHEME
        )

        # appropriate zoom in on north pole
        camera = Camera(
            eye=Eye(x=-0.1 / ZOOM_DEFAULT, y=-0.1 / ZOOM_DEFAULT, z=10 / ZOOM_DEFAULT)
        )

        # pick largest tick max value
        tick_mark = (
            abs(self.amplitude)
            if self.amplitude is not None
            else max(abs(vmin), abs(vmax))
        )

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                cmax=1 if settings.NORMALISE else tick_mark,
                cmid=0.5 if settings.NORMALISE else 0,
                cmin=0 if settings.NORMALISE else -tick_mark,
                colorbar=ColorBar(
                    x=0.93,
                    len=0.98,
                    nticks=2 if settings.NORMALISE else None,
                    tickfont=Tickfont(color="#666666", size=32),
                    tickformat=None if settings.NORMALISE else "+.1e",
                    tick0=None if settings.NORMALISE else -tick_mark,
                    dtick=None if settings.NORMALISE else tick_mark,
                ),
                colorscale=convert_colourscale(cmocean.cm.ice),
                lighting=Lighting(ambient=1),
                reversescale=True,
            )
        ]

        axis = dict(
            title="",
            showgrid=False,
            zeroline=False,
            ticks="",
            showticklabels=False,
            showbackground=False,
        )

        layout = Layout(
            scene=Scene(
                dragmode="orbit",
                camera=camera,
                xaxis=XAxis(axis),
                yaxis=YAxis(axis),
                zaxis=ZAxis(axis),
                annotations=self.annotations,
            ),
            margin=Margin(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig = Figure(data=data, layout=layout)

        # create html and open if auto_open is true
        html_filename = str(_fig_path / "html" / f"{self.filename}.html")

        py.plot(fig, filename=html_filename, auto_open=settings.AUTO_OPEN)

        # if save_fig is true then create png and pdf in their directories
        if settings.SAVE_FIG:
            for file_type in ["png", "pdf"]:
                logger.info(f"saving {file_type}")
                filename = str(_fig_path / file_type / f"{self.filename}.{file_type}")
                pio.write_image(fig, filename)

    @staticmethod
    def _setup_plot(
        f: np.ndarray,
        resolution: int,
        method: str = "MW",
        close: bool = True,
        parametric: bool = False,
        parametric_scaling: List[float] = [0.0, 0.5],
        color_range: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        function which creates the data for the matplotlib/plotly plot
        """
        if method == "MW_pole":
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        thetas, phis = ssht.sample_positions(resolution, Grid=True, Method=method)

        if thetas.size != f.size:
            raise Exception("Bandlimit L deos not match that of f")

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
            f_plot[f_plot == -1.56e30] = np.nan

        # Compute position scaling for parametric plot.
        if parametric:
            f_normalised = (f_plot - vmin / (vmax - vmin)) * parametric_scaling[
                1
            ] + parametric_scaling[0]

        # Close plot.
        if close:
            n_theta, n_phi = ssht.sample_shape(resolution, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, 0], axis=1)
            if parametric:
                f_normalised = np.insert(
                    f_normalised, n_phi, f_normalised[:, 0], axis=1
                )
            thetas = np.insert(thetas, n_phi, thetas[:, 0], axis=1)
            phis = np.insert(phis, n_phi, phis[:, 0], axis=1)

        # Compute location of vertices.
        if parametric:
            x, y, z = ssht.spherical_to_cart(f_normalised, thetas, phis)
        else:
            x, y, z = ssht.s2_to_cart(thetas, phis)

        return x, y, z, f_plot, vmin, vmax

    def _prepare_field(self, f: np.ndarray) -> np.ndarray:
        """
        boosts, forces plot type and then scales the field before plotting
        """
        return self._normalise_function(self._create_plot_type(self._boost_field(f)))

    def _boost_field(self, f: np.ndarray) -> np.ndarray:
        """
        inverts and then boosts the field before plotting
        """
        if not settings.UPSAMPLE:
            return f
        flm = ssht.forward(
            f, self.L, Reality=self.reality, Spin=self.spin, Method=SAMPLING_SCHEME
        )
        return invert_flm_boosted(
            flm, self.L, self.resolution, reality=self.reality, spin=self.spin
        )

    def _create_plot_type(self, f: np.ndarray) -> np.ndarray:
        """
        gets the given plot type of the field
        """
        logger.info(f"plotting type: '{self.plot_type}'")
        if self.plot_type == "abs":
            return np.abs(f)
        elif self.plot_type == "imag":
            return f.imag
        elif self.plot_type == "real":
            return f.real
        elif self.plot_type == "sum":
            return f.real + f.imag

    @staticmethod
    def _normalise_function(f: np.ndarray) -> np.ndarray:
        """
        normalise function between 0 and 1 for visualisation
        """
        if not settings.NORMALISE:
            return f
        elif (f == 0).all():
            # if all 0, set to 0
            return f + 0.5
        elif (f == f.max()).all():
            # if all non-zero, set to 1
            return f / f.max()
        else:
            # scale from [0, 1]
            return (f - f.min()) / f.ptp()
