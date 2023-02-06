from typing import Optional

from plotly.graph_objs import Layout
from plotly.graph_objs.layout import Margin, Scene
from plotly.graph_objs.layout.scene import Camera, XAxis, YAxis, ZAxis
from plotly.graph_objs.layout.scene.camera import Center, Eye

_axis = dict(
    title="",
    showgrid=False,
    zeroline=False,
    ticks="",
    showticklabels=False,
    showbackground=False,
)


def create_camera(
    x_eye: float,
    y_eye: float,
    z_eye: float,
    zoom: float,
    *,
    x_center: float = 0,
    y_center: float = 0,
    z_center: float = 0,
) -> Camera:
    """
    creates default camera view with a zoom factor
    """
    return Camera(
        eye=Eye(x=x_eye / zoom, y=y_eye / zoom, z=z_eye / zoom),
        center=Center(x=x_center, y=y_center, z=z_center),
    )


def create_layout(
    camera: Camera, *, annotations: Optional[list[dict]] = None
) -> Layout:
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
    fmin: float, fmax: float, *, amplitude: Optional[float] = None
) -> float:
    """
    creates tick mark to use when using a non-normalised plot
    """
    return amplitude if amplitude is not None else max(abs(fmin), abs(fmax))


def create_colour_bar(
    tick_mark: float,
    normalise: bool,
    *,
    bar_len: float = 0.94,
    bar_pos: float = 0.97,
    font_size: int = 38,
) -> dict:
    """
    default plotly colour bar
    """
    return dict(
        x=bar_pos,
        len=bar_len,
        nticks=2 if normalise else None,
        tickfont=dict(color="#666666", size=font_size),
        tickformat=None if normalise else "+.1e",
        tick0=None if normalise else -tick_mark,
        dtick=None if normalise else tick_mark,
    )