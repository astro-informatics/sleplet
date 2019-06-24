import sys
import os
import numpy as np
import cmocean
import plotly.offline as py
from plotly.graph_objs import Figure, Surface, Layout
from plotly.graph_objs.layout import Margin, Scene
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis
import plotly.io as pio
from fractions import Fraction
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
import scipy.special as sp
from typing import List, Tuple
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


class SiftingConvolution:
    def __init__(self, flm: np.array, flm_name: str, config: dict, glm: np.array=None, glm_name: str=None) -> None:
        self.auto_open = config['auto_open']
        self.flm_name = flm_name
        self.flm = flm
        self.glm = glm
        self.L = config['L']
        self.ncpu = config['ncpu']
        self.L_scipy_max = 86
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.method = config['sampling']
        self.reality = config['reality']
        self.resolution = self.calc_resolution(config)
        self.routine = config['routine']
        self.save_fig = config['save_fig']
        self.type = config['type']
        self.annotation = config['annotation']
        if self.glm is not None:
            self.glm_name = glm_name

    # -----------------------------------
    # ---------- flm functions ----------
    # -----------------------------------

    def rotation(self, flm: np.array, alpha: float, beta: float, gamma: float) -> np.ndarray:
        '''
        rotates given flm on the sphere by alpha/beta/gamma
        '''
        flm_rot = ssht.rotate_flms(
            flm, alpha, beta, gamma, self.L)
        return flm_rot

    def translation(self, flm: np.array) -> np.ndarray:
        '''
        translates given flm on the sphere by alpha/beta
        '''
        # numpy binary filename
        filename = os.path.join(
            self.location, 'npy', (f'trans_dd_L-{self.L}_'
                                   f'{self.filename_angle(self.alpha_pi_fraction,self.beta_pi_fraction)}'
                                   f'samp-{self.method}.npy'))

        # check if file of translated dirac delta already
        # exists otherwise calculate translated dirac delta
        # translation is slow for large L
        if os.path.exists(filename):
            glm = np.load(filename)
        else:
            glm = self.translate_dirac_delta(filename)

        # convolve with flm
        if self.flm_name == 'dirac_delta':
            flm_conv = glm
        else:
            flm_conv = self.convolution(flm, glm)
        return flm_conv

    def translate_dirac_delta(self, filename: str) -> np.ndarray:
        '''
        translates the dirac delta on the sphere to alpha/beta
        '''
        # initialise array
        flm_trans = np.zeros((self.L * self.L), dtype=complex)

        # scipy fails above L = 86
        if self.L < self.L_scipy_max + 1:
            L_max = self.L
        else:
            L_max = self.L_scipy_max

        # scipy method
        flm_trans = self.translate_dd_scipy(flm_trans, L_max)

        # choose method based on number of cores
        if self.ncpu == 1:
            flm_trans = self.translate_dd_serial(flm_trans)
        else:
            flm_trans = self.translate_dd_parallel(flm_trans)

        # save to speed up for future
        if filename is not None:
            np.save(filename, flm_trans)

        return flm_trans

    def convolution(self, flm: np.array, glm: np.array) -> np.ndarray:
        '''
        computes the sifting convolution of two arrays
        '''
        # translation/convolution are not real for general
        # function so turn off reality except for Dirac delta
        self.reality = False

        return flm * np.conj(glm)

    # ---------------------------------
    # ---------- translation ----------
    # ---------------------------------

    def translate_dd_scipy(self, flm: np.array, L: int) -> np.ndarray:
        '''
        scipy method to translate dirac delta up to L=86
        '''
        for ell in range(L):
            m = 0
            ind = ssht.elm2ind(ell, m)
            flm[ind] = sp.sph_harm(
                m, ell, -alpha, beta)
            for m in range(1, ell + 1):
                ind_pm = ssht.elm2ind(ell, m)
                ind_nm = ssht.elm2ind(ell, -m)
                flm[ind_pm] = sp.sph_harm(
                    m, ell, -alpha, beta)
                flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])
        return flm

    def translate_dd_serial(self, flm: np.array) -> np.ndarray:
        '''
        serial method to translate dirac delta - faster locally
        '''
        for ell in range(self.L_scipy_max, self.L):
            ind = ssht.elm2ind(ell, m=0)
            conj_pixel_val = self.calc_pixel_value(ind)
            flm[ind] = conj_pixel_val
            for m in range(1, ell + 1):
                print(ell, m)
                ind_pm = ssht.elm2ind(ell, m)
                ind_nm = ssht.elm2ind(ell, -m)
                conj_pixel_val = self.calc_pixel_value(ind_pm)
                flm[ind_pm] = conj_pixel_val
                flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])
        return flm

    def translate_dd_parallel(self, flm: np.array) -> np.ndarray:
        '''
        parallel method to translate dirac delta
        ideas come from
        https://jonasteuwen.github.io/numpy/python/multiprocessing/2017/01/07/multiprocessing-numpy-array.html
        '''
        # avoid strided arrays
        real = flm.real.copy()
        imag = flm.imag.copy()

        # create arrays to store final and intermediate steps
        result_r = np.ctypeslib.as_ctypes(real)
        result_i = np.ctypeslib.as_ctypes(imag)
        shared_array_r = sct.RawArray(result_r._type_, result_r)
        shared_array_i = sct.RawArray(result_i._type_, result_i)

        # ensure function declared before multiprocessing pool
        global func

        def func(chunk: List[int]) -> None:
            '''
            perform translation for real function using
            the conjugate symmetry for real signals
            '''
            # store real and imag parts separately
            tmp_r = np.ctypeslib.as_array(shared_array_r)
            tmp_i = np.ctypeslib.as_array(shared_array_i)

            # deal with chunk
            for ell in chunk:
                # m = 0 components
                ind = ssht.elm2ind(ell, m=0)
                conj_pixel_val = self.calc_pixel_value(ind)
                tmp_r[ind] = conj_pixel_val.real
                tmp_i[ind] = conj_pixel_val.imag

                # odd/even numbers
                for m in range(1, ell + 1):
                    print(ell, m)
                    ind_pm = ssht.elm2ind(ell, m)
                    ind_nm = ssht.elm2ind(ell, -m)
                    conj_pixel_val = self.calc_pixel_value(ind_pm)
                    # conjugate symmetry for real signals
                    tmp_r[ind_pm] = conj_pixel_val.real
                    tmp_i[ind_pm] = conj_pixel_val.imag
                    tmp_r[ind_nm] = (-1) ** m * tmp_r[ind_pm]
                    tmp_i[ind_nm] = (-1) ** (m + 1) * tmp_i[ind_pm]

        # split up L range to maximise effiency
        arr = np.arange(self.L_scipy_max, self.L)
        size = len(arr)
        arr[size // 2:size] = arr[size // 2:size][::-1]
        chunks = [np.sort(arr[i::self.ncpu]) for i in range(self.ncpu)]

        # initialise pool and apply function
        with mp.Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve real and imag components
        result_r = np.ctypeslib.as_array(shared_array_r)
        result_i = np.ctypeslib.as_array(shared_array_i)

        # combine results
        flm_trans = result_r + 1j * result_i

        return flm_trans

    # ----------------------------------------
    # ---------- plotting functions ----------
    # ----------------------------------------

    def plotly_plot(self, f: np.ndarray, filename: str, save_figure: bool) -> None:
        '''
        creates basic plotly plot rather than matplotlib
        '''
        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self.setup_plot(f)

        # appropriate zoom in on north pole
        zoom = 1.63
        camera = dict(
            eye=dict(x=-0.1 / zoom, y=-0.1 / zoom, z=2 / zoom)
        )

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                colorscale=self.cmocean_to_plotly('solar'),
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(
                    x=0.92,
                    len=0.98,
                    nticks=5,
                    tickfont=dict(
                        color='#666666',
                        size=32
                    )
                )
            )
        ]

        axis = dict(
            title='',
            showgrid=False,
            zeroline=False,
            ticks='',
            showticklabels=False
        )

        layout = Layout(
            scene=Scene(
                dragmode='orbit',
                camera=camera,
                xaxis=XAxis(axis),
                yaxis=YAxis(axis),
                zaxis=ZAxis(axis),
                annotations=self.annotations() if self.annotation else []
            ),
            margin=Margin(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = Figure(data=data, layout=layout)

        # if save_fig is true then print as png and pdf in their directories
        if save_figure:
            png_filename = os.path.join(
                self.location, 'figures', 'png', f'{filename}.png')
            pio.write_image(fig, png_filename)
            pdf_filename = os.path.join(
                self.location, 'figures', 'pdf', f'{filename}.pdf')
            pio.write_image(fig, pdf_filename)

        # create html and open if auto_open is true
        html_filename = os.path.join(
            self.location, 'figures', 'html', f'{filename}.html')
        py.plot(fig, filename=html_filename, auto_open=self.auto_open)

    def plot(self, alpha_pi_fraction: float=0.75, beta_pi_fraction: float=0.25, gamma_pi_fraction: float=0) -> None:
        '''
        master plotting method
        '''
        # setup
        gamma = gamma_pi_fraction * np.pi
        filename = f'{self.flm_name}_L-{self.L}_'

        # calculate nearest index of alpha/beta for translation
        self.calc_nearest_grid_point(alpha_pi_fraction, beta_pi_fraction)

        # test for plotting routine
        if self.routine == 'north':
            flm = self.flm
        elif self.routine == 'rotate':
            # adjust filename
            filename += (f'{self.routine}_'
                         f'{self.filename_angle(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)}')
            # rotate by alpha, beta
            flm = self.rotation(self.flm, self.alpha, self.beta, gamma)
        elif self.routine == 'translate':
            # adjust filename
            # don't add gamma if translation
            filename += (f'{self.routine}_'
                         f'{self.filename_angle(alpha_pi_fraction, beta_pi_fraction)}')
            # translate by alpha, beta
            flm = self.translation(self.flm)

        if self.glm is not None:
            # perform convolution
            flm = self.convolution(flm, self.glm)
            # adjust filename
            filename += f'convolved_{self.glm_name}_L-{self.L}_'

        # boost resolution
        if self.resolution != self.L:
            flm = self.resolution_boost(flm)

        # add sampling/resolution to filename
        filename += f'samp-{self.method}_res-{self.resolution}_'

        # inverse & plot
        f = ssht.inverse(flm, self.resolution,
                         Method=self.method, Reality=self.reality)

        # check for plotting type
        if self.type == 'real':
            plot = f.real
        elif self.type == 'imag':
            plot = f.imag
        elif self.type == 'abs':
            plot = abs(f)
        elif self.type == 'sum':
            plot = f.real + f.imag

        # do plot
        filename += self.type
        self.plotly_plot(plot, filename, self.save_fig)

    # --------------------------------------------------
    # ---------- translation helper functions ----------
    # --------------------------------------------------

    def calc_pixel_value(self, ind: int) -> complex:
        '''
        calculate the ylm(omega') which defines the translation
        '''
        # create Ylm corresponding to index
        ylm_harmonic = np.zeros(self.L * self.L, dtype=complex)
        ylm_harmonic[ind] = 1

        # convert Ylm from pixel to harmonic space
        ylm_pixel = ssht.inverse(ylm_harmonic, self.L, Method=self.method)

        # get value at pixel (i, j)
        ylm_omega = np.conj(ylm_pixel[self.pix_i, self.pix_j])

        return ylm_omega

    def calc_nearest_grid_point(self, alpha_pi_fraction: float=0, beta_pi_fraction: float=0) -> None:
        '''
        calculate nearest index of alpha/beta for translation
        this is due to calculating \omega' through the pixel
        values - the translation needs to be at the same position
        as the rotation such that the difference error is small
        '''
        thetas, phis = ssht.sample_positions(self.L, Method=self.method)
        self.pix_j = (np.abs(phis - alpha_pi_fraction * np.pi)).argmin()
        self.pix_i = (np.abs(thetas - beta_pi_fraction * np.pi)).argmin()
        self.alpha = phis[self.pix_j]
        self.beta = thetas[self.pix_i]
        self.alpha_pi_fraction = alpha_pi_fraction
        self.beta_pi_fraction = beta_pi_fraction

    # -----------------------------------------------
    # ---------- plotting helper functions ----------
    # -----------------------------------------------

    def annotations(self) -> List[dict]:
        # if north alter values to point at correct point
        if self.routine == 'north':
            x, y, z = 0, 0, 1
        else:
            x, y, z = ssht.s2_to_cart(self.beta, self.alpha)

        # initialise array and standard arrow
        annotation = []
        config = dict(arrowcolor='white', yshift=5)
        arrow = {**dict(x=x, y=y, z=z), **config}

        # various switch cases for annotation
        if self.flm_name.startswith('elongated_gaussian'):
            if self.routine == 'translate':
                annotation.append({**dict(x=-x, y=y, z=z), **config})
                annotation.append({**dict(x=x, y=-y, z=z), **config})
        elif self.flm_name == 'dirac_delta':
            if self.type != 'imag':
                annotation.append(arrow)
        elif 'gaussian' in self.flm_name:
            if self.routine != 'translate':
                if self.type != 'imag':
                    annotation.append(arrow)

        # if convolution then remove annotation
        if self.glm is not None:
            annotation = []
        return annotation

    @staticmethod
    def pi_in_filename(numerator: int, denominator: int) -> str:
        '''
        create filename for angle as multiple of pi
        '''
        # if whole number
        if denominator == 1:
            # if 1 * pi
            if numerator == 1:
                filename = 'pi'
            else:
                filename = f'{numerator}pi'
        else:
            filename = f'{numerator}pi{denominator}'
        return filename

    @staticmethod
    def get_angle_num_dem(angle_fraction: float) -> Tuple[int, int]:
        '''
        ger numerator and denominator for a given decimal
        '''
        angle = Fraction(angle_fraction).limit_denominator()
        return angle.numerator, angle.denominator

    @staticmethod
    def calc_resolution(config: dict) -> int:
        '''
        calculate appropriate resolution for given L
        '''
        if 'pow2_res2L' in config:
            exponent = config['pow2_res2L']
        else:
            if config['L'] == 1:
                exponent = 6
            elif config['L'] < 4:
                exponent = 5
            elif config['L'] < 8:
                exponent = 4
            elif config['L'] < 128:
                exponent = 3
            elif config['L'] < 512:
                exponent = 2
            elif config['L'] < 1024:
                exponent = 1
            else:
                exponent = 0
        resolution = config['L'] * 2 ** exponent

        return resolution

    @staticmethod
    def cmocean_to_plotly(colour, pl_entries: int=255) -> List[Tuple[float, str]]:
        '''
        converts cmocean colourscale to a plotly colourscale
        '''
        cmap = getattr(cmocean.cm, colour)

        h = 1 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
            pl_colorscale.append((k * h, f'rgb{(C[0], C[1], C[2])}'))

        return pl_colorscale

    def resolution_boost(self, flm: np.array) -> np.ndarray:
        '''
        calculates a boost in resoltion for given flm
        '''
        boost = self.resolution * self.resolution - self.L * self.L
        flm_boost = np.pad(flm, (0, boost), 'constant')
        return flm_boost

    def filename_angle(self, alpha_pi_fraction: float, beta_pi_fraction: float, gamma_pi_fraction: float=0) -> str:
        '''
        middle part of filename
        '''
        # get numerator/denominator for filename
        alpha_num, alpha_den = self.get_angle_num_dem(alpha_pi_fraction)
        beta_num, beta_den = self.get_angle_num_dem(beta_pi_fraction)
        gamma_num, gamma_den = self.get_angle_num_dem(gamma_pi_fraction)

        # if alpha = beta = 0
        if not alpha_num and not beta_num:
            filename = 'alpha-0_beta-0_'
        # if alpha = 0
        elif not alpha_num:
            filename = f'alpha-0_beta-{self.pi_in_filename(beta_num, beta_den)}_'
        # if beta = 0
        elif not beta_num:
            filename = f'alpha-{self.pi_in_filename(alpha_num, alpha_den)}_beta-0_'
        # if alpha != 0 && beta !=0
        else:
            filename = (f'alpha-{self.pi_in_filename(alpha_num, alpha_den)}_'
                        f'beta-{self.pi_in_filename(beta_num, beta_den)}_')

        # if rotation with gamma != 0
        if gamma_num:
            filename += f'gamma-{self.pi_in_filename(gamma_num, gamma_den)}_'
        return filename

    def setup_plot(self, f: np.ndarray, close: bool=True, parametric: bool=False, parametric_scaling: List[float]=[0.0, 0.5], color_range: List[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        '''
        function which creates the data for the matplotlib/plotly plot

        Arguments:
            f {function} -- inverse of flm

        Keyword Arguments:
            close {bool} -- if true the full sphere is plotted without a gap (default: {True})
            parametric {bool} -- the radius of the object at a certain point is defined by the function (default: {False})
            parametric_scaling {list} -- used if Parametric=True, defines the radius of the shape at a particular angle (default: {[0.0, 0.5]})
            color_range {list} -- if set saturates the color bar in that range, else the function min and max is used (default: {None})

        Raises:
            Exception -- if band limit L is not the same size as function f

        Returns:
            tuple -- values for the plotting
        '''
        if self.method == 'MW_pole':
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        (thetas, phis) = ssht.sample_positions(
            self.resolution, Method=self.method, Grid=True)

        if (thetas.size != f.size):
            raise Exception('Band-limit L deos not match that of f')

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
                self.resolution, Method=self.method)
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
