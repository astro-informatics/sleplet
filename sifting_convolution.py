import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import cm
import plotly.offline as py
from plotly.graph_objs import Figure, Surface, Layout
from plotly.graph_objs.layout import Scene
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


class SiftingConvolution(object):
    def __init__(self, L, resolution, fun=None):
        self.L = L
        self.resolution = resolution
        self.fun = fun

    @staticmethod
    def matplotlib_to_plotly(colour, pl_entries=255):
        '''
        converts matplotlib colourscale to a plotly colourscale

        Arguments:
            colour {string} -- matplotlib colour
        
        Keyword Arguments:
            pl_entries {bits} -- colour type (default: {255})
        
        Returns:
            plotly colour -- used in plotly plots
        '''

        cmap = cm.get_cmap(colour)

        h = 1.0 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

        return pl_colorscale

    def spherical_harmonic(self, ell, m):
        '''
        generates harmonic space representation of spherical harmonic
        
        Arguments:
            ell {int} -- current multipole value
            m {int} -- m <= |el|
        
        Returns:
            array -- square array shape: L x L
        '''

        ylm = np.zeros((self.L * self.L), dtype=complex)
        ind = ssht.elm2ind(ell, m)
        ylm[ind] = 1

        return ylm

    def north_pole(self, m_zero=False):
        '''
        calculates a given function on the north pole of the sphere
        
        Arguments:
            fun {function} -- the function to go on the north pole
        
        Keyword Arguments:
            m_zero {bool} -- whether m = 0 (default: {False})
        
        Returns:
            array -- new flm on the north pole
        '''

        def helper(self, flm, ell, m):
            '''
            calculates the value of flm at a particular value of ell and m
            
            Arguments:
                flm {array} -- initially array of zeros, gradually populated
                fun {function} -- the function to go on the sphere
                ell {int} -- the given value in the loop
                m {int} -- the given value in the loop
            
            Returns:
                array -- the flm after particular index has been set
            '''

            ind = ssht.elm2ind(ell, m)
            flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi)) * self.fun(ell, m)
            return flm

        # initiliase flm
        flm = np.zeros((self.resolution * self.resolution), dtype=complex)

        for ell in range(self.L):
            if not m_zero:
                for m in range(-ell, ell + 1):
                    flm = helper(self, flm, ell, m)
            else:
                flm = helper(self, flm, ell, m=0)

        return flm

    def sifting_convolution(self, flm, alpha, beta):
        '''
        applies the sifting convolution to a given flm
        
        Arguments:
            flm {array} -- square array shape: res x res
            alpha {float} -- the phi angle direction
            beta {float} -- the theta angle direction
        
        Returns:
            array -- the new flm after the convolution
        '''

        flm_conv = flm.copy()
        pix_i = ssht.phi_to_index(alpha, self.L)
        pix_j = ssht.theta_to_index(beta, self.L)

        for ell in range(self.L):
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                ylm_harmonic = self.spherical_harmonic(ell, m)
                ylm_real = ssht.inverse(ylm_harmonic, self.L)
                flm_conv[ind] = flm[ind] * ylm_real[pix_i, pix_j]

        return flm_conv

    def setup_plot(self, f, method='MW', close=True, parametric=False,
                   parametric_scaling=[0.0, 0.5], color_range=None):
        '''
        function which creates the data for the matplotlib/plotly plot
        
        Arguments:
            f {function} -- inverse of flm
        
        Keyword Arguments:
            method {str} -- sampling scheme (default: {'MW'})
            close {bool} -- if true the full sphere is plotted without a gap (default: {True})
            parametric {bool} -- the radius of the object at a certain point is defined by the function (default: {False})
            parametric_scaling {list} -- used if Parametric=True, defines the radius of the shape at a particular angle (default: {[0.0, 0.5]})
            color_range {list} -- if set saturates the color bar in that range, else the function min and max is used (default: {None})
        
        Raises:
            Exception -- if band limit L is not the same size as function f
        
        Returns:
            tuple -- values for the plotting
        '''

        # add ability to choose color bar min max
        # and sort out shapes of the plots

        if method == 'MW_pole':
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        (thetas, phis) = ssht.sample_positions(
            self.resolution, Method=method, Grid=True)

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
            f_normalised = (f_plot - vmin / (vmax - vmin)) * \
                parametric_scaling[1] + parametric_scaling[0]

        # % Close plot.
        if close:
            (n_theta, n_phi) = ssht.sample_shape(
                self.resolution, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, 0], axis=1)
            if parametric:
                f_normalised = np.insert(
                    f_normalised, n_phi, f_normalised[:, 0], axis=1)
            thetas = np.insert(thetas, n_phi, thetas[:, 0], axis=1)
            phis = np.insert(phis, n_phi, phis[:, 0], axis=1)

        # % Compute location of vertices.
        if parametric:
            (x, y, z) = ssht.spherical_to_cart(f_normalised, thetas, phis)
        else:
            (x, y, z) = ssht.s2_to_cart(thetas, phis)

        return x, y, z, f_plot, vmin, vmax

    def plotly_plot(self, f):
        '''
        creates basic plotly plot rather than matplotlib
        
        Arguments:
            f {function} -- inverse flm
        '''

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self.setup_plot(f)

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                colorscale=self.matplotlib_to_plotly('viridis'),
                cmin=vmin,
                cmax=vmax,
            )]

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

    def animation(self, flm, alphas, betas):
        xs, ys, zs, f_plots, vmins, vmaxs = \
            [], [], [], [], [], []

        angles = np.array([(a, b) for a in alphas for b in betas])

        for angle in angles:
            print(angle[0], angle[1])
            flm_conv = self.sifting_convolution(flm, angle[0], angle[1])
            f = ssht.inverse(flm_conv, self.resolution)
            x, y, z, f_plot, vmin, vmax = self.setup_plot(f.real)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            f_plots.append(f_plot)
            vmins.append(vmin)
            vmaxs.append(vmax)

        data = [
            Surface(
                x=xs[i],
                y=ys[i],
                z=zs[i],
                surfacecolor=f_plots[i],
                colorscale=self.matplotlib_to_plotly('viridis'),
                cmin=vmins[i],
                cmax=vmaxs[i],
                visible=False
            ) for i in range(len(angles))]
        curr_a_val = alphas.min()
        curr_b_val = betas.min()
        idx = np.where((angles == (curr_a_val, curr_b_val)).all(axis=1))[0][0]
        data[idx]['visible'] = True

        steps = []
        for c, angle in enumerate(angles):
            step = dict(
                method='restyle',
                args=['visible', [False] * len(angles)],
                label=str(angle),
            )
            step['args'][1][c] = True
            steps.append(step)

        # sliders = [
        #     # alpha
        #     dict(
        #         active=0,
        #         currentvalue=dict(
        #             prefix='$\\alpha$:',
        #         ),
        #         pad={"t": 0},
        #         steps=alpha_steps
        #     ),
        #     # beta
        #     dict(
        #         active=0,
        #         currentvalue=dict(
        #             prefix='$\\beta$:',
        #         ),
        #         pad={"t": 0},
        #         steps=beta_steps
        #     ),
        # ]

        sliders = [
            dict(
                active=0,
                currentvalue=dict(
                    prefix='(alpha/phi,beta/theta):',
                ),
                pad={"t": 0},
                steps=steps
            )
        ]

        axis = dict(title='')

        layout = Layout(
            scene=Scene(
                camera=dict(
                    eye=dict(x=1.25, y=-1.25, z=1.25)
                ),
                xaxis=XAxis(axis),
                yaxis=YAxis(axis),
                zaxis=ZAxis(axis)
            ),
            sliders=sliders
        )

        fig = Figure(data=data, layout=layout)

        py.plot(fig)
