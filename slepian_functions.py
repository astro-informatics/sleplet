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
        self,
        L: int,
        phi_min: int,
        phi_max: int,
        theta_min: int,
        theta_max: int,
        order: int = 0,
        double: bool = False,
    ):
        self.L = L
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        self.order = order
        self.phi_max = phi_max
        self.phi_max_is_default = phi_max == 360
        self.phi_min = phi_min
        self.phi_min_is_default = phi_min == 0
        self.theta_max = theta_max
        self.theta_max_is_default = theta_max == 180
        self.theta_min = theta_min
        self.theta_min_is_default = theta_min == 0
        self.is_polar_cap = (
            self.phi_min_is_default
            and self.phi_max_is_default
            and self.theta_min_is_default
            and not self.theta_max_is_default
        )
        self.is_polar_gap = self.is_polar_cap and double
        self.is_whole_sphere = (
            self.phi_min_is_default
            and self.phi_max_is_default
            and self.theta_min_is_default
            and self.theta_max_is_default
        )
        self.input_check()
        self.eigenvalues, self.eigenvectors = self.eigenproblem()

    # ----------------------------------------
    # ---------- D matrix functions ----------
    # ----------------------------------------

    def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        calculates the eigenvalues and eigenvectors of the D matrix
        """
        self.filename = ""
        if self.is_polar_cap:
            # polar cap/gap name addition
            if self.is_polar_gap:
                self.filename += "_gap"
            self.filename += f"_m{self.order}"
            # numpy binary filename
            binary = os.path.join(
                self.location,
                "npy",
                "slepian",
                "polar",
                f"D{self.filename_angle()}{self.filename}_L{self.L}",
            )
            spc = SlepianPolarCap(self.L, self.theta_max, binary, self.is_polar_gap)
            eigenvalues, eigenvectors = spc.eigenproblem(self.order)
        else:
            # numpy binary filename
            binary = os.path.join(
                self.location,
                "npy",
                "slepian",
                "lat_long",
                f"K{self.filename_angle()}{self.filename}_L{self.L}",
            )
            slll = SlepianLimitLatLong(
                self.L,
                self.phi_min,
                self.phi_max,
                self.theta_min,
                self.theta_max,
                binary,
            )
            eigenvalues, eigenvectors = slll.eigenproblem()
        return eigenvalues, eigenvectors

    # -----------------------------------------------
    # ---------- plotting helper functions ----------
    # -----------------------------------------------

    def annotations(self) -> List[dict]:
        """
        annotations for the plotly plot
        """
        annotation = []
        config = dict(arrowhead=6, ax=5, ay=5)
        # check if dealing with small polar cap
        if self.is_polar_cap and self.theta_max <= 45:
            ndots = 12
            theta = np.array(np.deg2rad(self.theta_max))
            for i in range(ndots):
                phi = np.array(2 * np.pi / ndots * (i + 1))
                x, y, z = ssht.s2_to_cart(theta, phi)
                annotation.append({**dict(x=x, y=y, z=z, arrowcolor="black"), **config})
            # check if dealing with polar gap
            if self.is_polar_gap:
                theta_bottom = np.array(np.pi - np.deg2rad(self.theta_max))
                for i in range(ndots):
                    phi = np.array(2 * np.pi / ndots * (i + 1))
                    x, y, z = ssht.s2_to_cart(theta_bottom, phi)
                    annotation.append(
                        {**dict(x=x, y=y, z=z, arrowcolor="white"), **config}
                    )
        # check if other region
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
                        annotation.append(
                            {**dict(x=x, y=y, z=z, arrowcolor="black"), **config}
                        )
        return annotation

    def filename_angle(self) -> str:
        """
        middle part of filename
        """
        # initialise filename
        filename = ""

        # if phi min is not default
        if not self.phi_min_is_default:
            filename += f"_pmin{int(self.phi_min)}"

        # if phi max is not default
        if not self.phi_max_is_default:
            filename += f"_pmax{int(self.phi_max)}"

        # if theta min is not default
        if not self.theta_min_is_default:
            filename += f"_tmin{int(self.theta_min)}"

        # if theta max is not default
        if not self.theta_max_is_default:
            filename += f"_tmax{int(self.theta_max)}"
        return filename

    # --------------------------
    # ---------- misc ----------
    # --------------------------

    def input_check(self) -> None:
        """
        check that inputs of class are valid
        """
        # check that angles are integers
        if not (
            self.phi_min.is_integer()
            and self.phi_max.is_integer()
            and self.theta_min.is_integer()
            and self.theta_max.is_integer()
        ):
            raise ValueError("angles for Slepian region should be integers")
        self.phi_min, self.phi_max, self.theta_min, self.theta_max = (
            int(self.phi_min),
            int(self.phi_max),
            int(self.theta_min),
            int(self.theta_max),
        )

        # order only important if Slepian polar cap
        if self.is_polar_cap:
            # test if order is an integer
            if not self.order.is_integer():
                raise ValueError(f"Slepian polar cap order should be an integer")
            self.order = int(self.order)

            # check order is in correct range
            if abs(self.order) >= self.L:
                raise ValueError(
                    f"Slepian polar cap order magnitude should be less than {self.L}"
                )

        # check that min/max theta values are correct
        if self.theta_min > self.theta_max:
            self.theta_min, self.theta_max = self.theta_max, self.theta_min
        elif self.theta_min == self.theta_max:
            raise ValueError("Invalid region.")

        # check that min/max phi values are correct
        if self.phi_min > self.phi_max:
            self.phi_min, self.phi_max = self.phi_max, self.phi_min
        elif self.phi_min == self.phi_max:
            raise ValueError("Invalid region.")
