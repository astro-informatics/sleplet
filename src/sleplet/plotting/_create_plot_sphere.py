import dataclasses
import logging

import cmocean
import numpy as np
import numpy.typing as npt
import plotly.graph_objs as go
import plotly.io as pio
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._plotly_methods
import sleplet._validation
import sleplet._vars
import sleplet.plot_methods
import sleplet.slepian.region

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class PlotSphere:
    """Create surface sphere plot via `plotly`."""

    f: npt.NDArray[np.complex_ | np.float_]
    """The field value sampled on the sphere."""
    L: int
    """The spherical harmonic bandlimit."""
    filename: str
    """The output filename of the plot."""
    _: dataclasses.KW_ONLY
    amplitude: float | None = None
    """Whether to customise the amplitude range of the colour bar."""
    annotations: list[dict[str, float | int]] = pydantic.Field(default_factory=list)
    """Whether to display any annotations on the surface plot or not."""
    normalise: bool = True
    """Whether to normalise the plot or not."""
    plot_type: str = "real"
    """Whether to display the `real`, `imag`, `abs` or `sum` value of the field."""
    reality: bool = False
    """Whether the given signal is real or not."""
    region: sleplet.slepian.region.Region | None = None
    """Whether to set the field values outside of a given region to zero."""
    spin: int = 0
    """Spin value."""
    upsample: bool = True
    """Whether to upsample the current field."""
    _resolution: int = pydantic.Field(default=0, init_var=False, repr=False)

    def __post_init__(self: typing_extensions.Self) -> None:
        self._resolution = (
            sleplet.plot_methods.calc_plot_resolution(self.L)
            if self.upsample
            else self.L
        )
        if self.upsample:
            self.filename += f"_res{self._resolution}"
        self.filename += f"_{self.plot_type}"
        if self.normalise:
            self.filename += "_norm"

    def execute(self: typing_extensions.Self) -> None:
        """Perform the plot."""
        f = self._prepare_field(self.f)

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self._setup_plot(
            f,
            self._resolution,
            method=sleplet._vars.SAMPLING_SCHEME,
        )

        if isinstance(self.region, sleplet.slepian.region.Region):
            # make plot area clearer
            f_plot = sleplet.plot_methods._set_outside_region_to_minimum(
                f_plot,
                self._resolution,
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
            go.Surface(
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
                lighting=go.surface.Lighting(ambient=1),
                reversescale=True,
            ),
        ]

        layout = sleplet._plotly_methods.create_layout(
            camera,
            annotations=self.annotations,
        )

        fig = go.Figure(data=data, layout=layout)

        msg = f"Opening: {self.filename}"
        _logger.info(msg)
        pio.show(fig, config={"toImageButtonOptions": {"filename": self.filename}})

    @staticmethod
    def _setup_plot(  # noqa: PLR0913
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
        """Create the data for the matplotlib/plotly plot."""
        if parametric_scaling is None:
            parametric_scaling = [0.0, 0.5]

        thetas, phis = ssht.sample_positions(resolution, Grid=True, Method=method)

        if thetas.size != f.size:
            msg = "Bandlimit L deos not match that of f"
            raise AttributeError(msg)

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
        self: typing_extensions.Self,
        f: npt.NDArray[np.complex_ | np.float_],
    ) -> npt.NDArray[np.float_]:
        """Boosts, forces plot type and then scales the field before plotting."""
        boosted_field = sleplet.plot_methods._boost_field(
            f,
            self.L,
            self._resolution,
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
