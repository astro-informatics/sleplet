import dataclasses
import logging

import cmocean
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import plotly.graph_objs as go
import plotly.io as pio
import pydantic
import typing_extensions

import sleplet._mask_methods
import sleplet._mesh_methods
import sleplet._plotly_methods
import sleplet._validation
import sleplet.meshes.mesh
import sleplet.plot_methods

_logger = logging.getLogger(__name__)

_MESH_CBAR_LEN = 0.95
_MESH_CBAR_FONT_SIZE = 32
_MESH_UNSEEN = -1e5  # kaleido bug


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class PlotMesh:
    """Create surface mesh plot via `plotly`."""

    mesh: sleplet.meshes.mesh.Mesh
    """The given mesh object."""
    filename: str
    """The output filename of the plot."""
    f: npt.NDArray[np.complex_ | np.float_]
    """The field value sampled on the mesh."""
    _: dataclasses.KW_ONLY
    amplitude: float | None = None
    """Whether to customise the amplitude range of the colour bar."""
    normalise: bool = True
    """Whether to normalise the plot or not."""
    region: bool = False
    """Whether to set the field values outside of the region to zero."""

    def __post_init__(self: typing_extensions.Self) -> None:
        if self.normalise:
            self.filename += "_norm"

    def execute(
        self: typing_extensions.Self,
        colour: mpl.colors.LinearSegmentedColormap = cmocean.cm.ice,
    ) -> None:
        """
        Perform the plot.

        Args:
            colour: From the `cmocean.cm` module
        """
        vmin, vmax = self.f.min(), self.f.max()
        f = self._prepare_field(self.f)

        if self.region:
            # make plot area clearer
            f = self._set_outside_region_to_minimum(f)

        # pick largest tick max value
        tick_mark = sleplet._plotly_methods.create_tick_mark(
            vmin,
            vmax,
            amplitude=self.amplitude,
        )

        data = [
            go.Mesh3d(
                x=self.mesh.vertices[:, 0],
                y=self.mesh.vertices[:, 2],
                z=self.mesh.vertices[:, 1],
                i=self.mesh.faces[:, 0],
                j=self.mesh.faces[:, 1],
                k=self.mesh.faces[:, 2],
                intensitymode="cell",
                intensity=f,
                cmax=1 if self.normalise else tick_mark,
                cmid=0.5 if self.normalise else 0,
                cmin=0 if self.normalise else -tick_mark,
                colorbar=sleplet._plotly_methods.create_colour_bar(
                    tick_mark,
                    normalise=self.normalise,
                    bar_len=_MESH_CBAR_LEN,
                    bar_pos=self.mesh._colourbar_pos,
                    font_size=_MESH_CBAR_FONT_SIZE,
                ),
                colorscale=sleplet.plot_methods._convert_colourscale(colour),
                lighting=go.mesh3d.Lighting(ambient=1),
                reversescale=True,
            ),
        ]

        layout = sleplet._plotly_methods.create_layout(self.mesh._camera_view)

        fig = go.Figure(data=data, layout=layout)

        msg = f"Opening: {self.filename}"
        _logger.info(msg)
        pio.show(fig, config={"toImageButtonOptions": {"filename": self.filename}})

    def _prepare_field(
        self: typing_extensions.Self,
        f: npt.NDArray[np.complex_ | np.float_],
    ) -> npt.NDArray[np.float_]:
        """Scales the field before plotting."""
        return sleplet.plot_methods._normalise_function(
            sleplet._mesh_methods.average_functions_on_vertices_to_faces(
                self.mesh.faces,
                f,
            ),
            normalise=self.normalise,
        )

    def _set_outside_region_to_minimum(
        self: typing_extensions.Self,
        f: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """
        For the Slepian region set the outside area to negative infinity
        hence it is clear we are only interested in the coloured region.
        """
        region_on_faces = sleplet._mask_methods.convert_region_on_vertices_to_faces(
            self.mesh,
        )
        return np.where(region_on_faces, f, _MESH_UNSEEN)
