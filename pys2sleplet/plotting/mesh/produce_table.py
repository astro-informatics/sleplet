from pys2sleplet.plotting.mesh.denoising_slepian_mesh import main
from pys2sleplet.utils.logger import logger

MESH_SNR_DICT = dict(cheetah=-8, dragon=-8, bird=-5, teapot=-3, cube=-7, homer=-5)
SIGMA = 1

if __name__ == "__main__":
    for mesh, snr in MESH_SNR_DICT.items():
        logger.info(f"\n\n\ndenoising {mesh}")
        main(mesh, snr, SIGMA)
