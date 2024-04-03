import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from denoising_slepian_mesh import main

MESH_SNR_DICT = {
    "cheetah": -8.64,
    "dragon": -8.12,
    "bird": -5.17,
    "teapot": -3.11,
    "cube": -7.33,
    "homer": -5,
}
SIGMA = 2

if __name__ == "__main__":
    for mesh, snr in MESH_SNR_DICT.items():
        print(f"\n\n\ndenoising {mesh}")
        main(mesh, snr, SIGMA)
