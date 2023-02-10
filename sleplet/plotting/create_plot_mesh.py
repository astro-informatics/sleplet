from dataclasses import KW_ONLY
from pathlib import Path

import cmocean
import numpy as np
import plotly.offline as py
from matplotlib.colors import LinearSegmentedColormap
from numpy import typing as npt
from plotly.graph_objs import Figure, Mesh3d
from plotly.graph_objs.mesh3d import Lighting
from pydantic.dataclasses import dataclass

from sleplet.meshes.classes.mesh import Mesh
from sleplet.utils.config import settings
from sleplet.utils.logger import logger
from sleplet.utils.mask_methods import convert_region_on_vertices_to_faces
from sleplet.utils.mesh_methods import average_functions_on_vertices_to_faces
from sleplet.utils.plot_methods import convert_colourscale, normalise_function
from sleplet.utils.plotly_methods import (
    create_colour_bar,
    create_layout,
    create_tick_mark,
)
from sleplet.utils.validation import Validation
from sleplet.utils.vars import MESH_CBAR_FONT_SIZE, MESH_CBAR_LEN, MESH_UNSEEN

_file_location = Path(__file__).resolve()
_fig_path = _file_location.parents[1] / "figures"


@dataclass(config=Validation)
class Plot:
    mesh: Mesh
    filename: str
    f: npt.NDArray
    _: KW_ONLY
    amplitude: float | None = None
    colour: LinearSegmentedColormap = cmocean.cm.ice
    normalise: bool = True
    region: bool = False

    def __post_init_post_parse__(self) -> None:
        if self.normalise:
            self.filename += "_norm"

    def execute(self) -> None:
        """
        creates 3d plotly mesh plot
        """
        vmin, vmax = self.f.min(), self.f.max()
        f = self._prepare_field(self.f)

        if self.region:
            # make plot area clearer
            f = self._set_outside_region_to_minimum(f)

        # pick largest tick max value
        tick_mark = create_tick_mark(vmin, vmax, amplitude=self.amplitude)

        data = [
            Mesh3d(
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
                colorbar=create_colour_bar(
                    tick_mark,
                    self.normalise,
                    bar_len=MESH_CBAR_LEN,
                    bar_pos=self.mesh.colourbar_pos,
                    font_size=MESH_CBAR_FONT_SIZE,
                ),
                colorscale=convert_colourscale(self.colour),
                lighting=Lighting(ambient=1),
                reversescale=True,
            )
        ]

        layout = create_layout(self.mesh.camera_view)

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

    def _prepare_field(self, f: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """
        scales the field before plotting
        """
        return normalise_function(
            average_functions_on_vertices_to_faces(self.mesh.faces, f), self.normalise
        )

    def _set_outside_region_to_minimum(
        self, f: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        for the Slepian region set the outisde area to negative infinity
        hence it is clear we are only interested in the coloured region
        """
        region_on_faces = convert_region_on_vertices_to_faces(self.mesh)
        return np.where(region_on_faces, f, MESH_UNSEEN)
