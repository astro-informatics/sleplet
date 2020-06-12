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
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis

from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import convert_colourscale
from pys2sleplet.utils.vars import SAMPLING_SCHEME, ZOOM_DEFAULT

_file_location = Path(__file__).resolve()
_fig_path = _file_location.parents[1] / "figures"


@dataclass
class Plot:
    f: np.ndarray = field(repr=False)
    resolution: int
    filename: str
    annotations: List[Dict] = field(default_factory=list, repr=False)

    def execute(self) -> None:
        """
        creates basic plotly plot rather than matplotlib
        """
        # get values from the setup
        f_scaled = (self.f - self.f.min()) / self.f.ptp()
        x, y, z, f_plot, vmin, vmax = self._setup_plot(
            f_scaled, self.resolution, method=SAMPLING_SCHEME
        )

        # appropriate zoom in on north pole
        camera = dict(
            eye=dict(x=-0.1 / ZOOM_DEFAULT, y=-0.1 / ZOOM_DEFAULT, z=10 / ZOOM_DEFAULT)
        )

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                colorscale=convert_colourscale(cmocean.cm.ice),
                reversescale=True,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(
                    x=0.84, len=0.98, nticks=2, tickfont=dict(color="#666666", size=32)
                ),
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

        # if save_fig is true then create png and pdf in their directories
        if config.SAVE_FIG:
            for file_type in ["png", "pdf"]:
                logger.info(f"saving {file_type}")
                filename = str(_fig_path / file_type / f"{self.filename}.{file_type}")
                pio.write_image(fig, filename)

        # create html and open if auto_open is true
        html_filename = str(_fig_path / "html" / f"{self.filename}.html")

        py.plot(fig, filename=html_filename, auto_open=config.AUTO_OPEN)

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

        if color_range is None:
            vmin = f_min
            vmax = f_max
        else:
            vmin = color_range[0]
            vmax = color_range[1]
            f_plot[f_plot < color_range[0]] = color_range[0]
            f_plot[f_plot > color_range[1]] = color_range[1]
            f_plot[f_plot == -1.56e30] = np.nan

        # % Compute position scaling for parametric plot.
        if parametric:
            f_normalised = (f_plot - vmin / (vmax - vmin)) * parametric_scaling[
                1
            ] + parametric_scaling[0]

        # % Close plot.
        if close:
            n_theta, n_phi = ssht.sample_shape(resolution, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, 0], axis=1)
            if parametric:
                f_normalised = np.insert(
                    f_normalised, n_phi, f_normalised[:, 0], axis=1
                )
            thetas = np.insert(thetas, n_phi, thetas[:, 0], axis=1)
            phis = np.insert(phis, n_phi, phis[:, 0], axis=1)

        # % Compute location of vertices.
        if parametric:
            x, y, z = ssht.spherical_to_cart(f_normalised, thetas, phis)
        else:
            x, y, z = ssht.s2_to_cart(thetas, phis)

        return x, y, z, f_plot, vmin, vmax
