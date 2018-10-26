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

    def translation(self, flm, alpha, beta):
        '''
        applies the translation operator to a given flm
        
        Arguments:
            flm {array} -- square array shape: res x res
            alpha {float} -- the phi angle direction
            beta {float} -- the theta angle direction
        
        Returns:
            array -- the new flm after the translation
        '''

        flm_conv = flm.copy()
        pix_i = ssht.theta_to_index(beta, self.L)
        pix_j = ssht.phi_to_index(alpha, self.L)

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

    def plotly_plot(self, f, html_name='temp-plot'):
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

        py.plot(fig, filename=html_name)

    def fun_plot(self, alpha, beta, f_type='north', gamma=0):
        # place function on the north pole on the sphere
        flm = self.north_pole(m_zero=True)
        filename = 'files/' + self.fun.func_name + '_' + f_type + '.html'

        if f_type == 'north':
            f = ssht.inverse(flm, self.resolution)
            self.plotly_plot(f.real, filename)
        elif f_type == 'rotate':
            flm_rot = ssht.rotate_flms(flm, alpha, beta, gamma, self.resolution)
            f_rot = ssht.inverse(flm_rot, self.resolution)
            self.plotly_plot(f_rot.real, filename)
        elif f_type == 'translate':
            flm_conv = self.translation(flm, alpha, beta)
            f_conv = ssht.inverse(flm_conv, self.resolution)
            self.plotly_plot(f_conv.real, filename)

    def flm_plot(self, alpha, beta, f_type='standard', gamma=0):
        # get the flm passed to the class
        flm = self.fun()
        filename = self.get_filename(f_type)

        if f_type == 'standard':
            f = ssht.inverse(flm, self.resolution)
            self.plotly_plot(f.real. filename)
        elif f_type == 'rotate':
            flm_rot = ssht.rotate_flms(flm, alpha, beta, gamma, self.resolution)
            f_rot = ssht.inverse(flm_rot, self.resolution)
            self.plotly_plot(f_rot.real, filename)
        elif f_type == 'translate':
            flm_conv = self.translation(flm, alpha, beta)
            f_conv = ssht.inverse(flm_conv, self.resolution)
            self.plotly_plot(f_conv.real, filename)
