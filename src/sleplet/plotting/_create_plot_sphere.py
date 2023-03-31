from dataclasses import KW_ONLY, field
from pathlib import Path

import cmocean
import numpy as np
import plotly.offline as py
import pyssht as ssht
from numpy import typing as npt
from plotly.graph_objs import Figure, Surface
from plotly.graph_objs.surface import Lighting
from pydantic.dataclasses import dataclass

import sleplet
import sleplet._plotly_methods
import sleplet._validation
import sleplet._vars
import sleplet.plot_methods
import sleplet.slepian.region

_fig_path = Path(__file__).resolve().parents[1] / "_figures"

MW_POLE_LENGTH = 2


@dataclass(config=sleplet._validation.Validation)
class PlotSphere:
    """creates surface sphere plot via `plotly`."""

    f: npt.NDArray[np.complex_ | np.float_]
    """TODO"""
    L: int
    """TODO"""
    filename: str
    """TODO"""
    _: KW_ONLY
    amplitude: float | None = None
    """TODO"""
    annotations: list[dict] = field(default_factory=list)
    """TODO"""
    normalise: bool = True
    """TODO"""
    plot_type: str = "real"
    """TODO"""
    reality: bool = False
    """TODO"""
    region: sleplet.slepian.region.Region | None = None
    """TODO"""
    spin: int = 0
    """TODO"""
    upsample: bool = True
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        self.resolution = (
            sleplet.plot_methods.calc_plot_resolution(self.L)
            if self.upsample
            else self.L
        )
        if self.upsample:
            self.filename += f"_res{self.resolution}"
        self.filename += f"_{self.plot_type}"
        if self.normalise:
            self.filename += "_norm"

    def execute(self) -> None:
        """TODO creates basic plotly plot rather than matplotlib."""
        f = self._prepare_field(self.f)

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self._setup_plot(
            f,
            self.resolution,
            method=sleplet._vars.SAMPLING_SCHEME,
        )

        if isinstance(self.region, sleplet.slepian.region.Region):
            # make plot area clearer
            f_plot = sleplet.plot_methods._set_outside_region_to_minimum(
                f_plot,
                self.resolution,
                self.region,
            )

        # appropriate zoom in on north pole
        camera = sleplet._plotly_methods.create_camera(-0.1, -0.1, 10, 7.88)

        # pick largest tick max value
        tick_mark = sleplet._plotly_methods.create_tick_mark(
            vmin,
            vmax,
            amplitude=self.amplitude,
        )

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                cmax=1 if self.normalise else tick_mark,
                cmid=0.5 if self.normalise else 0,
                cmin=0 if self.normalise else -tick_mark,
                colorbar=sleplet._plotly_methods.create_colour_bar(
                    tick_mark,
                    normalise=self.normalise,
                ),
                colorscale=sleplet.plot_methods._convert_colourscale(cmocean.cm.ice),
                lighting=Lighting(ambient=1),
                reversescale=True,
            ),
        ]

        layout = sleplet._plotly_methods.create_layout(
            camera,
            annotations=self.annotations,
        )

        fig = Figure(data=data, layout=layout)

        html_filename = str(_fig_path / "html" / f"{self.filename}.html")

        py.plot(fig, filename=html_filename)

        for file_type in {"png", "pdf"}:
            filename = str(_fig_path / file_type / f"{self.filename}.{file_type}")
            sleplet.logger.info(f"saving {filename}")
            fig.write_image(filename, engine="kaleido")

    @staticmethod
    def _setup_plot(
        f: npt.NDArray[np.float_],
        resolution: int,
        *,
        method: str = "MW",
        close: bool = True,
        parametric: bool = False,
        parametric_scaling: list[float] | None = None,
        color_range: list[float] | None = None,
    ) -> tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        float,
        float,
    ]:
        """function which creates the data for the matplotlib/plotly plot."""
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
            f_plot[f_plot == sleplet._vars.SPHERE_UNSEEN] = np.nan

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
                    f_normalised,
                    n_phi,
                    f_normalised[:, 0],
                    axis=1,
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

    def _prepare_field(
        self,
        f: npt.NDArray[np.complex_ | np.float_],
    ) -> npt.NDArray[np.float_]:
        """boosts, forces plot type and then scales the field before plotting."""
        boosted_field = sleplet.plot_methods._boost_field(
            f,
            self.L,
            self.resolution,
            reality=self.reality,
            spin=self.spin,
            upsample=self.upsample,
        )
        field_space = sleplet.plot_methods._create_plot_type(
            boosted_field,
            self.plot_type,
        )
        return sleplet.plot_methods._normalise_function(
            field_space,
            normalise=self.normalise,
        )
