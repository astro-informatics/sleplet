from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cmocean
import numpy as np
import plotly.io as pio
import plotly.offline as py
from plotly.graph_objs import Figure, Mesh3d
from plotly.graph_objs.layout.scene import Camera
from plotly.graph_objs.mesh3d import Lighting

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mask_methods import convert_region_on_vertices_to_faces
from pys2sleplet.utils.mesh_methods import average_functions_on_vertices_to_faces
from pys2sleplet.utils.plot_methods import convert_colourscale, normalise_function
from pys2sleplet.utils.plotly_methods import (
    create_colour_bar,
    create_layout,
    create_tick_mark,
)
from pys2sleplet.utils.vars import UNSEEN

_file_location = Path(__file__).resolve()
_fig_path = _file_location.parents[1] / "figures"
FUNCTIONS_IN_REGION: set[str] = {"region", "slepian"}


@dataclass
class Plot:
    mesh: Mesh
    f: np.ndarray = field(repr=False)
    filename: str
    camera_view: Camera
    colourbar_pos: float
    amplitude: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if settings.NORMALISE:
            self.filename += "_norm"

    def execute(self) -> None:
        """
        creates 3d plotly mesh plot
        """
        f = self._prepare_field(self.f)

        if set(self.filename.split("_")) & FUNCTIONS_IN_REGION:
            # make plot area clearer
            f = self._set_outside_region_to_minimum(f)

        # pick largest tick max value
        tick_mark = create_tick_mark(f.min(), f.max(), amplitude=self.amplitude)

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
                cmax=1 if settings.NORMALISE else tick_mark,
                cmid=0.5 if settings.NORMALISE else 0,
                cmin=0 if settings.NORMALISE else -tick_mark,
                colorbar=create_colour_bar(tick_mark, self.colourbar_pos),
                colorscale=convert_colourscale(cmocean.cm.ice),
                lighting=Lighting(ambient=1),
                reversescale=True,
            )
        ]

        layout = create_layout(self.camera_view)

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

    def _prepare_field(self, f: np.ndarray) -> np.ndarray:
        """
        scales the field before plotting
        """
        return normalise_function(
            average_functions_on_vertices_to_faces(self.mesh.faces, f)
        )

    def _set_outside_region_to_minimum(self, f: np.ndarray) -> np.ndarray:
        """
        for the Slepian region set the outisde area to negative infinity
        hence it is clear we are only interested in the coloured region
        """
        region_on_faces = convert_region_on_vertices_to_faces(self.mesh)
        return np.where(region_on_faces, f, UNSEEN)
