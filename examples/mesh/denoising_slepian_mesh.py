import argparse

import numpy as np
import numpy.typing as npt

import sleplet

B = 3
J_MIN = 2
MESHES = [
    "bird",
    "cheetah",
    "cube",
    "dragon",
    "homer",
    "teapot",
]
N_SIGMA = 1
NORMALISE = False
SNR_IN = -5


def _denoising_mesh_slepian(
    signal: sleplet.meshes.MeshSlepianField,
    noised_signal: sleplet.meshes.MeshSlepianField,
    mesh_slepian_wavelets: sleplet.meshes.MeshSlepianWavelets,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    """Denoising demo using Slepian wavelets."""
    # compute wavelet coefficients
    w = sleplet.wavelet_methods.slepian_wavelet_forward(
        noised_signal.coefficients,
        mesh_slepian_wavelets.wavelets,
        mesh_slepian_wavelets.mesh_slepian.N,
    )

    # compute wavelet noise
    sigma_j = sleplet.noise.compute_slepian_mesh_sigma_j(
        mesh_slepian_wavelets.mesh_slepian,
        signal.coefficients,
        mesh_slepian_wavelets.wavelets,
        snr_in,
    )

    # hard thresholding
    w_denoised = sleplet.noise.slepian_mesh_hard_thresholding(
        mesh_slepian_wavelets.mesh_slepian,
        w,
        sigma_j,
        n_sigma,
    )

    # wavelet synthesis
    f_p = sleplet.wavelet_methods.slepian_wavelet_inverse(
        w_denoised,
        mesh_slepian_wavelets.wavelets,
        mesh_slepian_wavelets.mesh_slepian.N,
    )

    # compute SNR
    sleplet.noise.compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return sleplet.slepian_methods.slepian_mesh_inverse(
        mesh_slepian_wavelets.mesh_slepian,
        f_p,
    )


def main(mesh_name: str, snr: float, sigma: int) -> None:
    """Denoising demo using Slepian wavelets."""
    print(f"SNR={snr}, n_sigma={sigma}")
    # setup
    mesh = sleplet.meshes.Mesh(mesh_name, zoom=True)

    # create map & noised map
    fun = sleplet.meshes.MeshSlepianField(mesh)
    fun_noised = sleplet.meshes.MeshSlepianField(mesh, noise=snr)

    # create wavelets
    smw = sleplet.meshes.MeshSlepianWavelets(mesh, B=B, j_min=J_MIN)

    # fix amplitude
    amplitude = np.abs(
        sleplet.slepian_methods.slepian_mesh_inverse(
            fun_noised.mesh_slepian,
            fun.coefficients,
        ),
    ).max()

    f = _denoising_mesh_slepian(fun, fun_noised, smw, snr, sigma)
    name = f"{mesh_name}_{snr}snr_{sigma}n_denoised"
    sleplet.plotting.PlotMesh(
        mesh,
        name,
        f,
        amplitude=amplitude,
        normalise=NORMALISE,
        region=True,
    ).execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="denoising")
    parser.add_argument(
        "function",
        type=str,
        choices=MESHES,
        help="mesh to plot",
        default="homer",
        const="homer",
        nargs="?",
    )
    parser.add_argument(
        "--noise",
        "-n",
        type=int,
        default=SNR_IN,
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=int,
        default=N_SIGMA,
    )
    args = parser.parse_args()
    main(args.function, args.noise, args.sigma)
