from plotting import Plotting
from slepian_limit_lat_long import SlepianLimitLatLong
from slepian_polar_cap import SlepianPolarCap
import numpy as np
import os
import sys
from typing import List, Tuple

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianFunctions:
    def __init__(
        self, phi_min: int, phi_max: int, theta_min: int, theta_max: int, config: dict
    ):
        self.L = config["L"]
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        self.phi_max = phi_max
        self.phi_max_is_default = phi_max == 360
        self.phi_min = phi_min
        self.phi_min_is_default = phi_min == 0
        self.plotting = Plotting(
            auto_open=config["auto_open"], save_fig=config["save_fig"]
        )
        self.resolution = self.plotting.calc_resolution(config["L"])
        self.theta_max = theta_max
        self.theta_max_is_default = theta_max == 180
        self.theta_min = theta_min
        self.theta_min_is_default = theta_min == 0
        self.angle_region_check()
        self.is_polar_cap = (
            self.phi_min_is_default
            and self.phi_max_is_default
            and self.theta_min_is_default
            and not self.theta_max_is_default
        )
        self.is_whole_sphere = (
            self.phi_min_is_default
            and self.phi_max_is_default
            and self.theta_min_is_default
            and self.theta_max_is_default
        )
        self.plotting.missing_key(config, "annotation", True)
        self.plotting.missing_key(config, "type", None)
        self.eigenvalues, self.eigenvectors = self.eigenproblem(config["order"])

    # ----------------------------------------
    # ---------- D matrix functions ----------
    # ----------------------------------------

    def eigenproblem(self, m) -> Tuple[np.ndarray, np.ndarray]:
        """
        calculates the eigenvalues and eigenvectors of the D matrix
        """
        self.filename = ""
        if self.is_polar_cap:
            spc = SlepianPolarCap(self.L, self.theta_max)
            eigenvalues, eigenvectors = spc.eigenproblem(m)
            if m < 0:
                self.filename = f"_m-n{abs(m)}"
            else:
                self.filename = f"_m-{m}"
        else:
            slll = SlepianLimitLatLong(
                self.L, self.phi_min, self.phi_max, self.theta_min, self.theta_max
            )
            eigenvalues, eigenvectors = slll.eigenproblem()
        return eigenvalues, eigenvectors

    # ----------------------------------------
    # ---------- plotting function -----------
    # ----------------------------------------

    def plot(self, rank: int) -> None:
        """
        master plotting method
        """
        # setup
        print(f"Eigenvalue {rank + 1}: {self.eigenvalues[rank]:e}")
        filename = f"slepian{self.filename_angle()}{self.filename}_L-{self.L}_rank-{rank + 1}_res-{self.resolution}_"
        flm = self.eigenvectors[rank]

        # boost resolution
        if self.resolution != self.L:
            flm = self.plotting.resolution_boost(flm, self.L, self.resolution)

        # inverse & plot
        f = ssht.inverse(flm, self.resolution, Method="MWSS")

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
        self.plotting.plotly_plot(f, self.resolution, filename, self.annotations())

    # -----------------------------------------------
    # ---------- plotting helper functions ----------
    # -----------------------------------------------

    def annotations(self) -> List[dict]:
        """
        annotations for the plotly plot
        """
        if self.plotting.annotation:
            annotation = []
            config = dict(arrowcolor="black", arrowhead=6, ax=5, ay=5)
            # check if dealing with small polar cap
            if self.is_polar_cap and self.theta_max <= 45:
                ndots = 12
                theta = np.array(np.deg2rad(self.theta_max))
                for i in range(ndots):
                    phi = np.array(2 * np.pi / ndots * (i + 1))
                    x, y, z = ssht.s2_to_cart(theta, phi)
                    annotation.append({**dict(x=x, y=y, z=z), **config})
            elif not self.is_whole_sphere:
                p1, p2, t1, t2 = (
                    np.array(np.deg2rad(self.phi_min)),
                    np.array(np.deg2rad(self.phi_max)),
                    np.array(np.deg2rad(self.theta_min)),
                    np.array(np.deg2rad(self.theta_max)),
                )
                p3, p4, t3, t4 = (
                    (p1 + 2 * p2) / 3,
                    (2 * p1 + p2) / 3,
                    (t1 + 2 * t2) / 3,
                    (2 * t1 + t2) / 3,
                )
                for t in [t1, t2, t3, t4]:
                    for p in [p1, p2, p3, p4]:
                        if not ((t == t3 or t == t4) and (p == p3 or p == p4)):
                            x, y, z = ssht.s2_to_cart(t, p)
                            annotation.append({**dict(x=x, y=y, z=z), **config})
        else:
            annotation = []
        return annotation

    def filename_angle(self) -> str:
        """
        middle part of filename
        """
        # initialise filename
        filename = ""

        # if phi min is not default
        if not self.phi_min_is_default:
            filename += f"_pmin-{self.phi_min}"

        # if phi max is not default
        if not self.phi_max_is_default:
            filename += f"_pmax-{self.phi_max}"

        # if theta min is not default
        if not self.theta_min_is_default:
            filename += f"_tmin-{self.theta_min}"

        # if theta max is not default
        if not self.theta_max_is_default:
            filename += f"_tmax-{self.theta_max}"
        return filename

    # --------------------------
    # ---------- misc ----------
    # --------------------------

    def angle_region_check(self):
        if self.theta_min > self.theta_max:
            self.theta_min, self.theta_max = self.theta_max, self.theta_min
        elif self.theta_min == self.theta_max:
            raise Exception("Invalid region.")

        if self.phi_min > self.phi_max:
            self.phi_min, self.phi_max = self.phi_max, self.phi_min
        elif self.phi_min == self.phi_max:
            raise Exception("Invalid region.")
