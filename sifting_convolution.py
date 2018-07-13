import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import cm, colors, colorbar, gridspec
import plotly.offline as py
from plotly.graph_objs import *

sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
import time


class SiftingConvolution:
    def __init__(self, L_comp, L_plot, gamma, beta, alpha):
        self.L_comp = L_comp
        self.L_plot = L_plot
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    @staticmethod
    def matplotlib_to_plotly(colour, pl_entries=255):
        cmap = cm.get_cmap(colour)

        h = 1.0 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

        return pl_colorscale

    def fill_flm(self, flm, fun, el, m):
        factor = np.sqrt((2 * el + 1) / (4 * np.pi))
        ind = ssht.elm2ind(el, m)
        ylm = self.spherical_harm(el, m)
        flm[ind] = factor * fun(el, m) * ylm[ind]

        return flm

    def spherical_harm(self, el, m):
        ylm = np.zeros((self.L_comp * self.L_comp), dtype=complex)
        ind = ssht.elm2ind(el, m)
        ylm[ind] = 1

        return ylm

    def sift_conv(self, flm):
        flm_conv = flm.copy()
        pix_i = ssht.phi_to_index(self.alpha, self.L_comp)
        pix_j = ssht.theta_to_index(self.beta, self.L_comp)

        for el in range(self.L_comp):
            for m in range(-el, el + 1):
                ind = ssht.elm2ind(el, m)
                ylm = self.spherical_harm(el, m)
                harm = ssht.inverse(ylm, self.L_comp)
                flm_conv[ind] = flm[ind] * harm[pix_i, pix_j]

        return flm_conv

    def north_pole(self, fun, m_zero=False):
        flm = np.zeros((self.L_plot * self.L_plot), dtype=complex)
        for el in range(self.L_comp):
            if not m_zero:
                for m in range(-el, el + 1):
                    flm = self.fill_flm(flm, fun, el, m)
            else:
                flm = self.fill_flm(flm, fun, el, m=0)

        return flm

    def dirac_delta(self):
        fun = lambda l, m: 1
        flm = self.north_pole(fun, m_zero=True)

        return flm

    def gaussian(self, sig=1):
        fun = lambda l, m: np.exp(-l * (l + 1)) / (2 * sig * sig)
        flm = self.north_pole(fun, m_zero=False)

        return flm

    def squashed_gaussian(self, sig=1):
        fun = lambda l, m: np.exp(m) * np.exp(-l * (l + 1)) / (2 * sig * sig)
        flm = self.north_pole(fun, m_zero=False)

        return flm

    def earth(self):
        matfile = os.path.join(
            os.environ['SSHT'], 'src', 'matlab', 'data',
            'EGM2008_Topography_flms_L0128')
        mat_contents = sio.loadmat(matfile)
        flm = np.ascontiguousarray(mat_contents['flm'][:, 0])

        return flm

    # Rotate spherical harmonic
    def rotate(self, flm):
        flm_rot = ssht.rotate_flms(
            flm, self.alpha, self.beta, self.gamma, self.L_plot)

        return flm_rot

    def dirac_delta_plot(self):
        flm = self.dirac_delta()
        f = ssht.inverse(flm, self.L_plot)
        flm_rot = self.rotate(flm)
        f_rot = ssht.inverse(flm_rot, self.L_plot)
        flm_conv = self.sift_conv(flm)
        f_conv = ssht.inverse(flm_conv, self.L_plot)
        # self.test_plot(f.real, parametric=False)
        # self.test_plot(f_rot.real)
        self.test_plot(f_conv.real)

    def gaussian_plot(self):
        flm = self.gaussian()
        f = ssht.inverse(flm, self.L_plot)
        flm_rot = self.rotate(flm)
        f_rot = ssht.inverse(flm_rot, self.L_plot)
        flm_conv = self.sift_conv(flm)
        f_conv = ssht.inverse(flm_conv, self.L_plot)
        # self.test_plot(f.real)
        # self.test_plot(f_rot.real)
        self.test_plot(f_conv.real)

    def squashed_gaussian_plot(self):
        flm = self.squashed_gaussian()
        f = ssht.inverse(flm, self.L_plot)
        flm_rot = self.rotate(flm)
        f_rot = ssht.inverse(flm_rot, self.L_plot)
        self.test_plot(f.real)
        self.test_plot(f_rot.real)

    def earth_plot(self):
        flm = self.earth()
        f = ssht.inverse(flm, self.L_plot)
        flm_rot = self.rotate(flm)
        f_rot = ssht.inverse(flm_rot, self.L_plot)
        self.test_plot(f.real)
        self.test_plot(f_rot.real)

    @staticmethod
    def matplotlib_to_plotly(colour, pl_entries=255):
        cmap = cm.get_cmap(colour)

        h = 1.0 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

        return pl_colorscale

    def test_plot(self, f, old_plot=False, method='MW', close=True, parametric=False,
                  parametric_scaling=[0.0, 0.5], output_file=None, show=True,
                  color_bar=True, units=None, color_range=None, axis=True):
        # add ability to choose color bar min max
        # and sort out shapes of the plots

        if method == 'MW_pole':
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        (thetas, phis) = ssht.sample_positions(self.L_plot, Method=method, Grid=True);

        if (thetas.size != f.size):
            raise Exception('Band limit L deos not match that of f')

        f_plot = f.copy()

        f_max = f_plot.max()
        f_min = f_plot.min()

        if color_range is None:
            vmin = f_min
            vmax = f_max
        else:
            vmin = color_range[0]
            vmax = color_range[1]
            f_plot[f_plot < color_range[0]] = color_range[0]
            f_plot[f_plot > color_range[1]] = color_range[1]
            f_plot[f_plot == -1.56E30] = np.nan

        # % Compute position scaling for parametric plot.
        if parametric:
            f_normalised = (f_plot - vmin / (vmax - vmin)) * parametric_scaling[1] + parametric_scaling[0]

        # % Close plot.
        if close:
            (n_theta, n_phi) = ssht.sample_shape(self.L_plot, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, 0], axis=1)
            if parametric:
                f_normalised = np.insert(f_normalised, n_phi, f_normalised[:, 0], axis=1)
            thetas = np.insert(thetas, n_phi, thetas[:, 0], axis=1)
            phis = np.insert(phis, n_phi, phis[:, 0], axis=1)

        # % Compute location of vertices.
        if parametric:
            (x, y, z) = ssht.spherical_to_cart(f_normalised, thetas, phis)
        else:
            (x, y, z) = ssht.s2_to_cart(thetas, phis)

        if old_plot:
            # % Plot.
            fig = plt.figure(figsize=plt.figaspect(1.1))
            gs = gridspec.GridSpec(2, 1, height_ratios=[10, 0.5])
            ax = fig.add_subplot(gs[0], projection='3d')
            ax_cbar = fig.add_subplot(gs[1])
            norm = colors.Normalize()
            surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(norm(f_plot)))
            if not axis:
                ax.set_axis_off()

            if color_bar:
                cmap = cm.jet
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                cb1 = colorbar.ColorbarBase(ax_cbar, cmap=cmap,
                                            norm=norm,
                                            orientation='horizontal')
                if units != None:
                    cb1.set_label(units)

            # output (to file and screan)
            if output_file != None:
                plt.savefig(output_file)
            if show:
                plt.show()

        data = Data([
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                colorscale=self.matplotlib_to_plotly('viridis'),
                cmin=vmin,
                cmax=vmax,
            )
        ])

        axis = dict(title='')

        layout = Layout(
            scene=Scene(
                camera=dict(
                    eye=dict(x=1.25, y=-1.25, z=1.25)
                ),
                xaxis=XAxis(axis),
                yaxis=YAxis(axis),
                zaxis=ZAxis(axis)
            )
        )

        fig = Figure(data=data, layout=layout)

        py.plot(fig)


if __name__ == '__main__':
    # Define parameters.
    L_comp = 2 ** 5
    L_plot = L_comp * 2 ** 3
    gamma = 0
    beta = np.pi / 4  # theta
    # beta = 0
    alpha = -np.pi / 4  # phi
    # alpha = 0

    sc = SiftingConvolution(L_comp, L_plot, gamma, beta, alpha)
    t0 = time.time()
    sc.dirac_delta_plot()
    # sc.gaussian_plot()
    t1 = time.time()
    print('TIME:', t1 - t0)
    # sc.squashed_gaussian_plot()
    # sc.earth_plot()

    # beta = np.linspace(0, np.pi, 9)
    # alpha = np.linspace(0, 2 * np.pi, 17)
    # pos = [(b, a) for b in beta for a in alpha]
    # f_conv = []
    # for p in pos:
    #     sc = SiftingConvolution(L_comp, L_plot, gamma, p[0], p[1])
    #     f_conv.append(sc.dirac_delta_plot())
    # frames = [dict(data=dict(f_conv))]

    # sliders=[
    #     # beta
    #     dict(
    #         active=0,
    #         currentvalue={'prefix': '$\\beta$: '},
    #         pad={'t': 50},
    #         steps=[dict(
    #             args=,
    #             method=
    #         ) for c, b in enumerate(beta)]
    #     ),
    #     # alpha
    #     dict(
    #         active=0,
    #         currentvalue={'prefix': '$\\alpha$: '},
    #         pad={'t': 50},
    #         steps=alpha
    #     )
    # ]
