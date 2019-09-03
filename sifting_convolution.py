from plotting import Plotting
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SiftingConvolution:
    def __init__(
        self,
        flm: np.ndarray,
        flm_name: str,
        config: dict,
        glm: np.ndarray = None,
        glm_name: str = None,
    ) -> None:
        self.annotations = (
            config["annotations"]
            if "annotations" in config and config["annotation"]
            else []
        )
        self.auto_open = config["auto_open"]
        self.flm_name = flm_name
        self.flm = flm
        self.glm = glm
        if self.glm is not None:
            self.glm_name = glm_name
        self.L = config["L"]
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        self.plotting = Plotting(
            auto_open=config["auto_open"], save_fig=config["save_fig"]
        )
        self.reality = config["reality"]
        self.save_fig = config["save_fig"]
        self.resolution = self.plotting.calc_resolution(config["L"])
        self.plotting.missing_key(config, "routine", None)
        self.plotting.missing_key(config, "type", None)

    # -----------------------------------
    # ---------- flm functions ----------
    # -----------------------------------

    def rotation(
        self, flm: np.ndarray, alpha: float, beta: float, gamma: float
    ) -> np.ndarray:
        """
        rotates given flm on the sphere by alpha/beta/gamma
        """
        flm_rot = ssht.rotate_flms(flm, alpha, beta, gamma, self.L)
        return flm_rot

    def translation(self, flm: np.ndarray) -> np.ndarray:
        """
        translates given flm on the sphere by alpha/beta
        """
        # numpy binary filename
        filename = os.path.join(
            self.location,
            "npy",
            "trans_dirac",
            (
                f"trans_dd_L{self.L}_"
                f"{self.filename_angle(self.alpha_pi_fraction,self.beta_pi_fraction)}.npy"
            ),
        )

        # check if file of translated dirac delta already
        # exists otherwise calculate translated dirac delta
        if os.path.exists(filename):
            glm = np.load(filename)
        else:
            glm = np.conj(ssht.create_ylm(self.beta, self.alpha, self.L))
            glm = glm.reshape(glm.size)
            # save to speed up for future
            np.save(filename, glm)

        # convolve with flm
        if self.flm_name == "dirac_delta":
            flm_conv = glm
        else:
            flm_conv = self.convolution(flm, glm)
        return flm_conv

    def convolution(self, flm: np.ndarray, glm: np.ndarray) -> np.ndarray:
        """
        computes the sifting convolution of two arrays
        """
        # translation/convolution are not real for general
        # function so turn off reality except for Dirac delta
        self.reality = False

        return flm * np.conj(glm)

    # ----------------------------------------
    # ---------- plotting function -----------
    # ----------------------------------------

    def plot(
        self,
        alpha_pi_fraction: float = 0.75,
        beta_pi_fraction: float = 0.25,
        gamma_pi_fraction: float = 0,
    ) -> None:
        """
        master plotting method
        """
        # setup
        gamma = gamma_pi_fraction * np.pi
        filename = f"{self.flm_name}_L{self.L}_"

        # calculate nearest index of alpha/beta for translation
        self.calc_nearest_grid_point(alpha_pi_fraction, beta_pi_fraction)

        # test for plotting routine
        if self.plotting.routine == "north":
            flm = self.flm
        elif self.plotting.routine == "rotate":
            # adjust filename
            filename += (
                f"{self.plotting.routine}_"
                f"{self.filename_angle(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)}_"
            )
            # rotate by alpha, beta
            flm = self.rotation(self.flm, self.alpha, self.beta, gamma)
        elif self.plotting.routine == "translate":
            # adjust filename
            # don't add gamma if translation
            filename += (
                f"{self.plotting.routine}_"
                f"{self.filename_angle(alpha_pi_fraction, beta_pi_fraction)}_"
            )
            # translate by alpha, beta
            flm = self.translation(self.flm)

        if self.glm is not None:
            # perform convolution
            flm = self.convolution(flm, self.glm)
            # adjust filename
            filename += f"convolved_{self.glm_name}_L{self.L}_"

        # boost resolution
        if self.resolution != self.L:
            flm = self.plotting.resolution_boost(flm, self.L, self.resolution)

        # add resolution to filename
        filename += f"res{self.resolution}_"

        # inverse & plot
        f = ssht.inverse(flm, self.resolution, Reality=self.reality, Method="MWSS")

        # check for plotting type
        if self.plotting.type == "real":
            f = f.real
        elif self.plotting.type == "imag":
            f = f.imag
        elif self.plotting.type == "abs":
            f = np.abs(f)
        elif self.plotting.type == "sum":
            f = f.real + f.imag

        # do plot
        filename += self.plotting.type
        self.plotting.plotly_plot(
            f, self.resolution, filename, annotations=self.annotations
        )

    # --------------------------------------------------
    # ---------- translation helper function ----------
    # --------------------------------------------------

    def calc_nearest_grid_point(
        self, alpha_pi_fraction: float = 0, beta_pi_fraction: float = 0
    ) -> None:
        """
        calculate nearest index of alpha/beta for translation
        this is due to calculating omega' through the pixel
        values - the translation needs to be at the same position
        as the rotation such that the difference error is small
        """
        thetas, phis = ssht.sample_positions(self.L, Method="MWSS")
        pix_j = (np.abs(phis - alpha_pi_fraction * np.pi)).argmin()
        pix_i = (np.abs(thetas - beta_pi_fraction * np.pi)).argmin()
        self.alpha = phis[pix_j]
        self.beta = thetas[pix_i]
        self.alpha_pi_fraction = alpha_pi_fraction
        self.beta_pi_fraction = beta_pi_fraction

    # -----------------------------------------------
    # ---------- plotting helper functions ----------
    # -----------------------------------------------

    def filename_angle(
        self,
        alpha_pi_fraction: float,
        beta_pi_fraction: float,
        gamma_pi_fraction: float = 0,
    ) -> str:
        """
        middle part of filename
        """
        # get numerator/denominator for filename
        alpha_num, alpha_den = self.plotting.get_angle_num_dem(alpha_pi_fraction)
        beta_num, beta_den = self.plotting.get_angle_num_dem(beta_pi_fraction)
        gamma_num, gamma_den = self.plotting.get_angle_num_dem(gamma_pi_fraction)

        # if alpha = beta = 0
        if not alpha_num and not beta_num:
            filename = "alpha0_beta0"
        # if alpha = 0
        elif not alpha_num:
            filename = f"alpha0_beta{self.plotting.pi_in_filename(beta_num, beta_den)}"
        # if beta = 0
        elif not beta_num:
            filename = (
                f"alpha{self.plotting.pi_in_filename(alpha_num, alpha_den)}_beta0"
            )
        # if alpha != 0 && beta !=0
        else:
            filename = (
                f"alpha{self.plotting.pi_in_filename(alpha_num, alpha_den)}"
                f"_beta{self.plotting.pi_in_filename(beta_num, beta_den)}"
            )

        # if rotation with gamma != 0
        if gamma_num:
            filename += f"gamma{self.plotting.pi_in_filename(gamma_num, gamma_den)}"
        return filename
