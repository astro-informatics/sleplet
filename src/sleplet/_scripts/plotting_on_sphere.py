import argparse
import logging

import numpy as np
import numpy.typing as npt

import pyssht as ssht

import sleplet._class_lists
import sleplet._mask_methods
import sleplet._string_methods
import sleplet.functions.coefficients
import sleplet.harmonic_methods
import sleplet.plot_methods
import sleplet.plotting._create_plot_sphere
import sleplet.slepian_methods

_logger = logging.getLogger(__name__)


_ALPHA_DEFAULT = 0.75
_ANNOTATION_COLOUR = "gold"
_ARROW_STYLE = {
    "arrowhead": 0,
    "arrowside": "start",
    "ax": 4,
    "ay": 4,
    "startarrowsize": 0.5,
    "startarrowhead": 6,
}
_BETA_DEFAULT = 0.125


def valid_maps(map_name: str) -> str:
    """Check if valid map."""
    if map_name in sleplet._string_methods.convert_classes_list_to_snake_case(
        sleplet._class_lists.MAPS_LM,
    ):
        return map_name
    msg = f"{map_name} is not a valid map to convolve"
    raise ValueError(msg)


def valid_plotting(func_name: str) -> str:
    """Check if valid function."""
    if func_name in sleplet._string_methods.convert_classes_list_to_snake_case(
        sleplet._class_lists.COEFFICIENTS,
    ):
        return func_name
    msg = f"{func_name} is not a valid function to plot"
    raise ValueError(msg)


def read_args() -> argparse.Namespace:
    """Read args from the command line."""
    parser = argparse.ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "function",
        type=valid_plotting,
        choices=sleplet._string_methods.convert_classes_list_to_snake_case(
            sleplet._class_lists.COEFFICIENTS,
        ),
        help="function to plot on the sphere",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=_ALPHA_DEFAULT,
        help=f"alpha/phi pi fraction - defaults to {_ALPHA_DEFAULT}",
    )
    parser.add_argument("--bandlimit", "-L", type=int, default=16, help="bandlimit")
    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=_BETA_DEFAULT,
        help=f"beta/theta pi fraction - defaults to {_BETA_DEFAULT}",
    )
    parser.add_argument(
        "--convolve",
        "-c",
        type=valid_maps,
        default=None,
        choices=sleplet._string_methods.convert_classes_list_to_snake_case(
            sleplet._class_lists.MAPS_LM,
        ),
        help="glm to perform sifting convolution with i.e. flm x glm*",
    )
    parser.add_argument(
        "--extra_args",
        "-e",
        type=int,
        nargs="+",
        help="list of extra args for functions",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        type=float,
        default=0,
        help="gamma pi fraction - defaults to 0 - rotation only",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        nargs="?",
        default="north",
        const="north",
        choices=["north", "rotate", "translate"],
        help="plotting routine: defaults to north",
    )
    parser.add_argument("--noise", "-n", type=int, help="the SNR_IN of the noise level")
    parser.add_argument(
        "--outline",
        "-o",
        action="store_false",
        help="flag which removes any annotation",
    )
    parser.add_argument(
        "--perspective",
        "-p",
        type=str,
        nargs="?",
        default="south_america",
        const="south_america",
        choices=["africa", "south_america"],
        help="view of Earth: defaults to 'south_america'",
    )
    parser.add_argument(
        "--region",
        "-r",
        action="store_true",
        help="flag which masks the function for a region",
    )
    parser.add_argument(
        "--smoothing",
        "-s",
        type=int,
        help="the scaling of the sigma in Gaussian smoothing of the Earth",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        nargs="?",
        default="real",
        const="real",
        choices=["abs", "real", "imag", "sum"],
        help="plotting type: defaults to real",
    )
    parser.add_argument(
        "--unnormalise",
        "-u",
        action="store_true",
        help="flag turns off normalisation for plot",
    )
    parser.add_argument(
        "--unzeropad",
        "-z",
        action="store_true",
        help="flag turns off upsampling for plot",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=sleplet.__version__,
    )
    return parser.parse_args()


def plot(  # noqa: PLR0913
    f: sleplet.functions.coefficients.Coefficients,
    g: sleplet.functions.coefficients.Coefficients | None,
    *,
    alpha_pi_frac: float,
    beta_pi_frac: float,
    gamma_pi_frac: float,
    annotations: bool,
    normalise: bool,
    method: str,
    plot_type: str,
    upsample: bool,
    earth_view: str,
    amplitude: float | None,
) -> None:
    """Master plotting method."""
    filename = f.name
    coefficients = f.coefficients

    # turn off annotation if needed
    msg = f"annotations on: {annotations}"
    _logger.info(msg)
    annotation = []

    # Shannon number for Slepian coefficients
    shannon = f.slepian.N if hasattr(f, "slepian") else None

    msg = f"plotting method: '{method}'"
    _logger.info(msg)
    match method:
        case "rotate":
            coefficients, filename = _rotation_helper(
                f,
                filename,
                alpha_pi_frac,
                beta_pi_frac,
                gamma_pi_frac,
            )
        case "translate":
            coefficients, filename, trans_annotation = _translation_helper(
                f,
                filename,
                alpha_pi_frac,
                beta_pi_frac,
                shannon,
            )

            # annotate translation point
            if annotations:
                annotation.append(trans_annotation)

    if g is not None:
        coefficients, filename = _convolution_helper(
            f,
            g,
            coefficients,
            shannon,
            filename,
        )

    # rotate plot of Earth
    if "earth" in filename:
        match earth_view:
            case "africa":
                coefficients = sleplet.harmonic_methods.rotate_earth_to_africa(
                    coefficients,
                    f.L,
                )
                filename += "_africa"
            case "south_america":
                coefficients = sleplet.harmonic_methods.rotate_earth_to_south_america(
                    coefficients,
                    f.L,
                )

    # get field value
    field = sleplet.plot_methods._coefficients_to_field_sphere(f, coefficients)

    # do plot
    sleplet.plotting._create_plot_sphere.PlotSphere(
        field,
        f.L,
        filename,
        amplitude=amplitude,
        annotations=annotation,
        normalise=normalise,
        plot_type=plot_type,
        reality=f.reality,
        region=f.region if hasattr(f, "slepian") else None,
        spin=f.spin,
        upsample=upsample,
    ).execute()


def _rotation_helper(
    f: sleplet.functions.coefficients.Coefficients,
    filename: str,
    alpha_pi_frac: float,
    beta_pi_frac: float,
    gamma_pi_frac: float,
) -> tuple[npt.NDArray[np.complex_], str]:
    """Perform the rotation specific steps."""
    msg = (
        "angles: (alpha, beta, gamma) = "
        f"({alpha_pi_frac}, {beta_pi_frac}, {gamma_pi_frac})",
    )
    _logger.info(msg)
    filename += (
        "_rotate_"
        f"{sleplet._string_methods.filename_angle(alpha_pi_frac, beta_pi_frac, gamma_pi_frac)}"  # noqa: E501
    )

    # calculate angles
    alpha, beta = sleplet.plot_methods._calc_nearest_grid_point(
        f.L,
        alpha_pi_frac,
        beta_pi_frac,
    )
    gamma = gamma_pi_frac * np.pi

    # rotate by alpha, beta, gamma
    coefficients = f.rotate(alpha, beta, gamma=gamma)
    return coefficients, filename


def _translation_helper(
    f: sleplet.functions.coefficients.Coefficients,
    filename: str,
    alpha_pi_frac: float,
    beta_pi_frac: float,
    shannon: int | None,
) -> tuple[npt.NDArray[np.complex_ | np.float_], str, dict[str, float | int]]:
    """Perform the translation specific steps."""
    msg = f"angles: (alpha, beta) = ({alpha_pi_frac}, {beta_pi_frac})"
    _logger.info(msg)
    # don't add gamma if translation
    filename += (
        "_translate_"
        f"{sleplet._string_methods.filename_angle(alpha_pi_frac, beta_pi_frac)}"
    )

    # calculate angles
    alpha, beta = sleplet.plot_methods._calc_nearest_grid_point(
        f.L,
        alpha_pi_frac,
        beta_pi_frac,
    )

    # translate by alpha, beta
    coefficients = f.translate(alpha, beta, shannon=shannon)

    # annotate translation point
    x, y, z = ssht.s2_to_cart(beta, alpha)
    annotation = {
        "x": x,
        "y": y,
        "z": z,
        "arrowcolor": _ANNOTATION_COLOUR,
    } | _ARROW_STYLE
    return coefficients, filename, annotation


def _convolution_helper(
    f: sleplet.functions.coefficients.Coefficients,
    g: sleplet.functions.coefficients.Coefficients,
    coefficients: npt.NDArray[np.complex_ | np.float_],
    shannon: int | None,
    filename: str,
) -> tuple[npt.NDArray[np.complex_ | np.float_], str]:
    """Perform the convolution specific steps."""
    g_coefficients = (
        sleplet.slepian_methods.slepian_forward(f.L, f.slepian, flm=g.coefficients)
        if hasattr(f, "slepian")
        else g.coefficients
    )
    coefficients = f.convolve(g_coefficients, coefficients, shannon=shannon)

    filename += f"_convolved_{g.name}"
    return coefficients, filename


def main() -> None:
    args = read_args()

    mask = sleplet._mask_methods.create_default_region() if args.region else None

    f = sleplet._class_lists.COEFFICIENTS[
        sleplet._string_methods.convert_classes_list_to_snake_case(
            sleplet._class_lists.COEFFICIENTS,
        ).index(args.function)
    ](
        args.bandlimit,
        extra_args=args.extra_args,
        region=mask,
        noise=args.noise if args.noise is not None else None,
        smoothing=args.smoothing if args.smoothing is not None else None,
    )

    g = (
        sleplet._class_lists.MAPS_LM[
            sleplet._string_methods.convert_classes_list_to_snake_case(
                sleplet._class_lists.MAPS_LM,
            ).index(args.convolve)
        ](args.bandlimit)
        if isinstance(args.convolve, str)
        else None
    )

    # custom amplitude for noisy plots
    amplitude = sleplet.plot_methods.compute_amplitude_for_noisy_sphere_plots(f)

    plot(
        f,
        g,
        alpha_pi_frac=args.alpha,
        beta_pi_frac=args.beta,
        gamma_pi_frac=args.gamma,
        annotations=args.outline,
        normalise=not args.unnormalise,
        method=args.method,
        plot_type=args.type,
        upsample=not args.unzeropad,
        earth_view=args.perspective,
        amplitude=amplitude,
    )


if __name__ == "__main__":
    main()
