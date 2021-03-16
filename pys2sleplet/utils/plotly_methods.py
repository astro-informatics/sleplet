from typing import Optional

from plotly.graph_objs import Layout
from plotly.graph_objs.layout import Margin, Scene
from plotly.graph_objs.layout.scene import Camera, XAxis, YAxis, ZAxis
from plotly.graph_objs.layout.scene.camera import Eye

from pys2sleplet.utils.config import settings

_axis = dict(
    title="",
    showgrid=False,
    zeroline=False,
    ticks="",
    showticklabels=False,
    showbackground=False,
)


def create_camera(x: float, y: float, z: float, zoom: float) -> Camera:
    """
    creates default camera view with a zoom factor
    """
    return Camera(eye=Eye(x=x / zoom, y=y / zoom, z=z / zoom))


def create_layout(camera: Camera, annotations: Optional[list[dict]] = None) -> Layout:
    """
    default plotly layout
    """
    return Layout(
        scene=Scene(
            dragmode="orbit",
            camera=camera,
            xaxis=XAxis(_axis),
            yaxis=YAxis(_axis),
            zaxis=ZAxis(_axis),
            annotations=annotations,
        ),
        margin=Margin(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )


def create_tick_mark(
    fmin: float, fmax: float, amplitude: Optional[float] = None
) -> float:
    """
    creates tick mark to use when using a non-normalised plot
    """
    return amplitude if amplitude is not None else max(abs(fmin), abs(fmax))


def create_colour_bar(tick_mark: float, bar_pos: float) -> dict:
    """
    default plotly colour bar
    """
    return dict(
        x=bar_pos,
        len=0.98,
        nticks=2 if settings.NORMALISE else None,
        tickfont=dict(color="#666666", size=32),
        tickformat=None if settings.NORMALISE else "+.1e",
        tick0=None if settings.NORMALISE else -tick_mark,
        dtick=None if settings.NORMALISE else tick_mark,
    )
