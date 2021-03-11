from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cmocean
import numpy as np
import plotly.io as pio
import plotly.offline as py
from plotly.graph_objs import Figure, Mesh3d
from plotly.graph_objs.mesh3d import Lighting

from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import (
    convert_colourscale,
    create_plot_type,
    normalise_function,
)
from pys2sleplet.utils.plotly_methods import (
    create_camera,
    create_colour_bar,
    create_layout,
    create_tick_mark,
)

_file_location = Path(__file__).resolve()
_fig_path = _file_location.parents[1] / "figures"


@dataclass
class Plot:
    vertices: np.ndarray = field(repr=False)
    triangles: np.ndarray = field(repr=False)
    f: np.ndarray = field(repr=False)
    filename: str
    amplitude: Optional[float] = field(default=None, repr=False)
    plot_type: str = field(default="real", repr=False)
    annotations: List[Dict] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.filename += f"_{self.plot_type}"
        if settings.NORMALISE:
            self.filename += "_norm"

    def execute(self) -> None:
        """
        creates 3d plotly mesh plot
        """
        f = self._prepare_field(self.f)

        # appropriate zoom in on north pole
        camera = create_camera(0, -1, 10, 6)

        # pick largest tick max value
        tick_mark = create_tick_mark(f.min(), f.max(), amplitude=self.amplitude)

        data = [
            Mesh3d(
                x=-self.vertices[:, 0],
                y=self.vertices[:, 1],
                z=-self.vertices[:, 2],
                i=self.triangles[:, 0],
                j=self.triangles[:, 1],
                k=self.triangles[:, 2],
                intensity=f,
                cmax=1 if settings.NORMALISE else tick_mark,
                cmid=0.5 if settings.NORMALISE else 0,
                cmin=0 if settings.NORMALISE else -tick_mark,
                colorbar=create_colour_bar(tick_mark, 0.8),
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
            for file_type in ["png", "pdf"]:
                logger.info(f"saving {file_type}")
                filename = str(_fig_path / file_type / f"{self.filename}.{file_type}")
                pio.write_image(fig, filename)

    def _prepare_field(self, f: np.ndarray) -> np.ndarray:
        """
        forces plot type and then scales the field before plotting
        """
        field_space = create_plot_type(f, self.plot_type)
        return normalise_function(field_space)
