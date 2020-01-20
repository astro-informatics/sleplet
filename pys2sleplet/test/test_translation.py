from unittest import TestCase

import numpy as np
from dynaconf import settings

from pys2sleplet.flm.kernels.dirac_delta import DiracDelta
from pys2sleplet.plotting.create_plot import Plot


class TestTranslation(TestCase):
    def setUp(self):
        self.alpha_pi_frac = 0.75
        self.beta_pi_frac = 0.125
        self.L = settings.L
        self.plot = settings.TEST_PLOTS

    def test_dirac_delta_rotate_translate(self) -> None:
        """
        test to ensure that rotation and translation
        give the same result for the Dirac delta
        """
        # rotation
        dd = DiracDelta(self.L)
        dd.rotate(self.alpha_pi_frac, self.beta_pi_frac)
        flm_rot = dd.multipole
        f_rot = dd.field

        # translation
        dd = DiracDelta(self.L)
        dd.translate(self.alpha_pi_frac, self.beta_pi_frac)
        flm_trans = dd.multipole
        f_trans = dd.field

        # calculate difference
        flm_diff = flm_rot - flm_trans
        f_diff = f_rot - f_trans

        # perform test
        np.testing.assert_allclose(flm_rot, flm_trans, atol=1e-14)
        np.testing.assert_allclose(f_rot, f_trans, rtol=1e-5)
        print("Translation/rotation difference max error:", np.max(np.abs(flm_diff)))

        if self.plot:
            filename = f"{dd.name}_L{self.L}_diff_rot_trans_res{dd.resolution}"
            Plot(f_diff.real, dd.resolution, filename).execute()
