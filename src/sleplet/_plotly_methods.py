import typing

import plotly.graph_objs as go

_axis = {
    "title": "",
    "showgrid": False,
    "zeroline": False,
    "ticks": "",
    "showticklabels": False,
    "showbackground": False,
}


def create_camera(  # noqa: PLR0913
    x_eye: float,
    y_eye: float,
    z_eye: float,
    zoom: float,
    *,
    x_center: float = 0,
    y_center: float = 0,
    z_center: float = 0,
) -> go.layout.scene.Camera:
    """Create default camera view with a zoom factor."""
    return go.layout.scene.Camera(
        eye=go.layout.scene.camera.Eye(x=x_eye / zoom, y=y_eye / zoom, z=z_eye / zoom),
        center=go.layout.scene.camera.Center(x=x_center, y=y_center, z=z_center),
    )


def create_layout(
    camera: go.layout.scene.Camera,
    *,
    annotations: list[dict[str, float | int]] | None = None,
) -> go.Layout:
    """Create the default plotly layout."""
    return go.Layout(
        scene=go.layout.Scene(
            dragmode="orbit",
            camera=camera,
            xaxis=go.layout.scene.XAxis(_axis),
            yaxis=go.layout.scene.YAxis(_axis),
            zaxis=go.layout.scene.ZAxis(_axis),
            annotations=annotations,
        ),
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )


def create_tick_mark(
    fmin: float,
    fmax: float,
    *,
    amplitude: float | None = None,
) -> float:
    """Create tick mark to use when using a non-normalised plot."""
    return amplitude if amplitude is not None else max(abs(fmin), abs(fmax))


def create_colour_bar(
    tick_mark: float,
    *,
    normalise: bool,
    bar_len: float = 0.94,
    bar_pos: float = 0.97,
    font_size: int = 38,
) -> dict[str, typing.Any]:
    """Create the default plotly colour bar."""
    return {
        "x": bar_pos,
        "len": bar_len,
        "nticks": 2 if normalise else None,
        "tickfont": {"color": "#666666", "size": font_size},
        "tickformat": None if normalise else "+.1e",
        "tick0": None if normalise else -tick_mark,
        "dtick": None if normalise else tick_mark,
    }
