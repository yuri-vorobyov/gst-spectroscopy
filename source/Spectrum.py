import numpy as np
import matplotlib.pyplot as plt
from sg_smooth.smoothing import smSG_bisquare
from scipy.interpolate import interp1d
import os.path


class RTPair:
    """
    Container for a pair of R and T spectra measured in one experiment using the same detector.
    """

    COLORS = {
        'R': '#1f77b5',
        'T': '#fd8114'
    }

    DETECTORS = {
        'Si': {
            'type': 'VIS',
            'limits': (600, 925)
        },
        'CaF2 MCT': {
            'type': 'NIR',
            'limits': (910, 2500)  # limited by 1737F substrate
        },
        'InGaAs': {
            'type': 'NIR',
            'limits': (840, 2400)
        }
    }

    def __init__(self, w, R, T, detector=None):
        """
        Parameters
        ----------
        w : array-like
            Array of wavelengths in nanometers.
        R : array_like
            Reflectance spectra.
        T : array_like
            Transmittance spectra.
        detector : str or None
            Detector type which was used for the spectra acquisition.
        """
        # Check correctness of input arrays.
        if not (len(w) == len(R) == len(T)):
            raise Exception('Input arrays lengths must be equal.')
        # Check if the detector type provided is supported.
        if detector:
            if detector not in RTPair.DETECTORS.keys():
                raise Exception(f'"{detector}" detector is not supported.')

        # Save data.
        self.w = w
        self.R = R
        self.T = T

        # Also in the form of single array (original data --- should remain untouched).
        self._data = np.column_stack((self.w, self.R, self.T))

        # Strip according to detector limits.
        if detector:
            self.strip(*RTPair.DETECTORS[detector]['limits'])

        # Smoothed version
        self.sw = None
        self.sR = None
        self.sT = None

    @classmethod
    def from_files(cls, R, T, detector=None):
        """
        Factory method to instantiate `RTPair` from text files with the spectra.
        Parameters
        ----------
        R : str
            File path to the R spectrum.
        T : str
            File path to the T spectrum.
        detector : str or None
            Detector type which was used for the spectra acquisition.
        """
        # Check file existence.
        if not os.path.exists(R):
            raise Exception(f'"{R}" cannot be found!')
        if not os.path.exists(T):
            raise Exception(f'"{T}" cannot be found!')
        # Load the data.
        r = np.loadtxt(R, skiprows=1, dtype=np.float64)
        t = np.loadtxt(T, skiprows=1, dtype=np.float64)
        # Check is wave-number scales equal to each other.
        if not np.allclose(r[:, 0], t[:, 0], rtol=1e-6):
            raise Exception('Looks like R and T are from different data sets --- wave-number scales are different.')
        # Save input data.
        return cls(1e7 / ((r[:, 0] + t[:, 0]) / 2), r[:, 1], t[:, 1], detector)

    @property
    def e(self):
        return 1239.842 / self.w

    @property
    def se(self):
        return 1239.842 / self.sw

    def strip(self, wl_min, wl_max):
        """
        Strip the wavelength scale. It is the initial spectra which are getting stripped, so this method could be
        invoked several times and will still provide correct results.

        Parameters
        ----------
        wl_min : float
            Minimum wavelength in nm.
        wl_max : float
            Maximum wavelength in nm.
        """
        data = self._data[(self._data[:, 0] > wl_min) * (self._data[:, 0] < wl_max)]
        self.w = data[:, 0]
        self.R = data[:, 1]
        self.T = data[:, 2]

    def calc_smoothed(self, w, n):
        """
        Calculate (update) filtered spectra.

        Parameters
        ----------
        w : int
            Radius of smoothing window (number of points).
        n : int
            Order of approximating polynomial.

        Returns
        -------

        """
        self.sw, self.sR, _ = smSG_bisquare(self.w, self.R, w, n, extend=False)
        _, self.sT, _ = smSG_bisquare(self.w, self.T, w, n, extend=False)

    def plot(self, scale='wavelength', title=''):
        """
        Plot the spectrum.

        Parameters
        ----------
        scale : str
            Either `wavelength` or `energy`.
        title : str
            Optional title.
        """
        if scale not in {'wavelength', 'energy'}:
            raise Exception('`scale` support only "wavelength" or "energy"')

        plt.style.use('style.mplstyle')
        plt.rcParams['savefig.directory'] = '.'
        fig, ax_T = plt.subplots(1, 1)
        fig.canvas.manager.set_window_title(title)
        ax_R = ax_T.twinx()
        ax_T.set_title(title)
        ax_T.set_xlabel({'wavelength': 'Wavelength (nm)',
                         'energy': 'Photon energy (eV)'}[scale])
        ax_T.set_ylabel('T')
        ax_R.set_ylabel('R')

        x = {'wavelength': self.w, 'energy': self.e}[scale]
        l_t, = ax_T.plot(x, self.T, c=RTPair.COLORS['T'], alpha=0.7, label='T')
        l_r, = ax_R.plot(x, self.R, c=RTPair.COLORS['R'], alpha=0.7, label='R')

        ax_T.legend(handles=(l_t, l_r), loc='best')

        plt.show(block=True)


class Spectrum:
    """
    Spectrum is a collection of several individual spectra — obtained using different detectors at some particular
    sample temperature.

    Water vapor has several absorption bands, according to literature: at 1.38, 1.87, 2.7, 6.3 um. In out experimental
    spectra first three contribute significantly:
        1340 – 1495 nm
        1785 – 1970 nm
        2500 – 2900 nm
    """

    PRINT_HEADER = False
    AUTO_SMOOTH = False
    AUTO_CORRECT = False
    WAVELENGTH_MIN = {
        'VIS': 630,  # nm
        'NIR': 910,  # nm, for CaF2 MCT
        'MIR': 1350  # nm
    }
    WAVELENGTH_MAX = {
        'VIS': 1160,  # nm
        'NIR': 2500,  # nm, for CaF2 MCT, limited by 1737F substrate
        'MIR': 6650  # nm
    }
    COLORS = {
        'R': '#1f77b5',
        'T': '#fd8114',
    }
    SMOOTHING_WINDOW_RADIUS = {
        'VIS': 20,
        'NIR': 30,
        'MIR': 150
    }
    SMOOTHING_POLY_ORDER = {
        'VIS': 3,
        'NIR': 3,
        'MIR': 3
    }
    WATER_BANDS = {
        'A': (1340, 1495),
        'B': (1785, 1970),
        'C': (2500, 2900)
    }

    def __init__(self, VIS_R='', VIS_T='', VIS_detector='',
                       NIR_R='', NIR_T='', NIR_detector='',
                       MIR_R='', MIR_T='', MIR_detector='',
                       temperature='RT'):
        """
        Parameters
        ----------
        VIS_R : str
            Filepath to the R spectrum obtained by VIS detector.
        VIS_T : str
            Filepath to the T spectrum obtained by VIS detector.
        VIS_detector : str
            VIS detector type.
        NIR_R : str
            Filepath to the R spectrum obtained by NIR detector.
        NIR_T : str
            Filepath to the T spectrum obtained by NIR detector.
        NIR_detector : str
            NIM detector type.
        MIR_R : str
            Filepath to the R spectrum obtained by MIR detector.
        MIR_T : str
            Filepath to the T spectrum obtained by MIR detector.
        MIR_detector : str
            MIR detector type.
        temperature : numeric or 'RT'
            Temperature of sample when the spectrum was measured.
        """

        # Check for consistency --- R and T spectra should be provided in pairs.
        if (bool(VIS_R) != bool(VIS_T)) or (bool(NIR_R) != bool(NIR_T)) or (bool(MIR_R) != bool(MIR_T)):
            raise Exception('R and T spectra go in pairs! Check input arguments.')

        # If necessary to do so, print headers to make sure there is no error in locating files.
        if Spectrum.PRINT_HEADER:
            if VIS_R:
                print(f'VIS R: {Spectrum.__file_header(VIS_R)}')
            if VIS_T:
                print(f'VIS T: {Spectrum.__file_header(VIS_T)}')
            if NIR_R:
                print(f'NIR R: {Spectrum.__file_header(NIR_R)}')
            if NIR_T:
                print(f'NIR T: {Spectrum.__file_header(NIR_T)}')
            if MIR_R:
                print(f'MIR R: {Spectrum.__file_header(MIR_R)}')
            if MIR_T:
                print(f'MIR T: {Spectrum.__file_header(MIR_T)}')

        # Containers for spectra data.
        self.VIS = None
        self.NIR = None
        self.MIR = None
        self.FULL = None

        self.detectors = set()

        # Load raw data, convert it to the internal representation and save for later use
        if VIS_R:
            self.detectors.add('VIS')
            self.VIS = RTPair.from_files(VIS_R, VIS_T, VIS_detector)
        if NIR_R:
            self.detectors.add('NIR')
            self.NIR = RTPair.from_files(NIR_R, NIR_T, NIR_detector)
        if MIR_R:
            self.detectors.add('MIR')
            self.MIR = RTPair.from_files(MIR_R, MIR_T, MIR_detector)

        # Save value of temperature
        self.temperature = temperature

        if Spectrum.AUTO_CORRECT:
            self.calculate_corrected(kind='linear')

    @staticmethod
    def __assert_detector(detector):
        if detector not in {'VIS', 'NIR', 'MIR'}:
            raise Exception('`detector` should be one of "VIS", "NIR", or "MIR"')

    def __assert_detector_in_use(self, detector):
        if detector not in self.detectors:
            raise Exception(f'No data provided for {detector}, only {self.detectors} are given.')

    def plot(self, spectra='raw', title=''):
        """
        Plot full spectrum: both R and T for all provided detectors.

        Parameters
        ----------
        spectra : str
            What data to plot. One of "raw", "smoothed", "stitched".
        title : str
            Optional title for the figure.
        """
        if spectra not in {'raw', 'smoothed', 'stitched'}:
            raise Exception('`spectra` should be "raw", or "smoothed", or "stitched".')

        plt.style.use('style.mplstyle')
        plt.rcParams['savefig.directory'] = '.'
        fig, ax_T = plt.subplots(1, 1)
        ax_R = ax_T.twinx()
        fig.canvas.manager.set_window_title(title)
        ax_T.set_title(title)
        ax_T.set_xlabel('Wavelength (nm)')
        ax_T.set_ylabel('T')
        ax_R.set_ylabel('R')

        kwargs = dict(alpha=0.65, lw=1.4)

        if spectra == 'stitched':
            ax_T.plot(self.FULL.w, self.FULL.T, c=Spectrum.COLORS['T'], **kwargs)
            ax_R.plot(self.FULL.w, self.FULL.R, c=Spectrum.COLORS['R'], **kwargs)
            plt.show(block=True)
            return

        legend_t, legend_r = [], []
        if self.VIS is not None:
            if spectra == 'raw':
                x, t, r = self.VIS.w, self.VIS.T, self.VIS.R
            else:
                x, t, r = self.VIS.sw, self.VIS.sT, self.VIS.sR
            vis_t, = ax_T.plot(x, t, c=Spectrum.COLORS['T'], **kwargs)
            vis_r, = ax_R.plot(x, r, c=Spectrum.COLORS['R'], **kwargs)
            legend_t.append(vis_t)
            legend_r.append(vis_r)
        if self.NIR is not None:
            if spectra == 'raw':
                x, t, r = self.NIR.w, self.NIR.T, self.NIR.R
            else:
                x, t, r = self.NIR.sw, self.NIR.sT, self.NIR.sR
            nir_t, = ax_T.plot(x, t, c=Spectrum.COLORS['T'], **kwargs)
            nir_r, = ax_R.plot(x, r, c=Spectrum.COLORS['R'], **kwargs)
            legend_t.append(nir_t)
            legend_r.append(nir_r)
        if self.MIR is not None:
            if spectra == 'raw':
                x, t, r = self.MIR.w, self.MIR.T, self.MIR.R
            else:
                x, t, r = self.MIR.sw, self.MIR.sT, self.MIR.sR
            mir_t, = ax_T.plot(x, t, c=Spectrum.COLORS['T'], **kwargs)
            mir_r, = ax_R.plot(x, r, c=Spectrum.COLORS['R'], **kwargs)
            legend_t.append(mir_t)
            legend_r.append(mir_r)

        legend_t = tuple(legend_t)
        legend_r = tuple(legend_r)
        ax_T.legend([legend_t, legend_r], ['T', 'R'])
        plt.show(block=True)

    @staticmethod
    def __file_header(fname):
        """Returns first line of a text file (hopefully with spectrum info)."""
        with open(fname, mode='r', encoding='utf8') as fs:
            return fs.readline().rstrip()

    @staticmethod
    def __correct(left_x, left_y, right_x, right_y, kind='uniform'):
        """
        Calculate corrected spectrum.

        Parameters
        ----------
        left_x, left_y : array-like
            Left, standard, spectrum.
        right_x, right_y: array-like
            Right, to be corrected, spectrum.
        kind : str
            Type of correction algorithm.

        Returns
        -------
        dict
            Corrected version of the right_y spectrum together with the stitched spectrum.
        """
        # Interpolate for the same set of wavelengths.
        w0, w1 = right_x[0], left_x[-1]
        # Find all the points within those limits.
        ii_left, ii_right = left_x >= w0, right_x <= w1
        wl_left, wl_right = left_x[ii_left], right_x[ii_right]
        # Interpolation points are taken from the part with the lesser number of points (improve performance a bit). In
        # principle, one may use any vector `w`, however, in the general case two calls to interpolation function are
        # needed.
        if len(wl_left) < len(wl_right):
            w, f = wl_left, left_y[ii_left]
            g = interp1d(wl_right, right_y[ii_right], 'cubic', assume_sorted=True)(wl_left)
        else:
            w, g = wl_right, right_y[ii_right]
            f = interp1d(wl_left, left_y[ii_left], 'cubic', assume_sorted=True)(wl_right)

        # Calculate corrected version of the right spectrum.
        A = (g * g).sum()
        D = (f * g).sum()
        if kind == 'uniform':
            b = D / A
            right_y_corr = right_y * b
            g_corr = g * b  # for transition region computation
        else:
            B = (w * g * g).sum()
            C = (w * w * g * g).sum()
            E = (w * f * g).sum()
            a0 = (C * D - E * B) / (C * A - B * B)
            a1 = (E * A - D * B) / (C * A - B * B)
            right_y_corr = (a0 + a1 * right_x) * right_y
            g_corr = (a0 + a1 * right_x) * g  # for transition region computation

        # Calculate transition region for the same wavelength points `w`. Again, it is not necessary to use the same set
        # of wavelengths here --- using `w` just leads to slightly faster calculation speed.
        tr = np.linspace(1.0, 0.0, len(w)) * f + \
             np.linspace(0.0, 1.0, len(w)) * g_corr

        # Finally, assemble the stitched spectrum.
        ii_left, ii_right = left_x < w0, right_x > w1
        stitched_x = np.concatenate((left_x[ii_left], w, right_x[ii_right]))
        stitched_y = np.concatenate((left_y[ii_left], tr, right_y_corr[ii_right]))

        return dict(right_y_corr=right_y_corr, stitched_x=stitched_x, stitched_y=stitched_y)

    def calculate_corrected(self, kind='uniform'):
        # Check for correctness.
        if self.detectors == {'VIS', 'MIR'}:
            raise Exception('Cannot MIR using VIS --- NIR is needed.')
        if len(self.detectors) == 1:
            raise Exception('Only one spectrum is provided — nothing to correct.')

        # Perform correction for two-detector case.
        if len(self.detectors) == 2:
            # Alias those two spectra.
            if self.detectors == {'VIS', 'NIR'}:
                left, right = self.VIS, self.NIR
            else:
                left, right = self.NIR, self.MIR

            # Calculate corrected spectra.
            R_corr = Spectrum.__correct(left.w, left.R, right.w, right.R, kind)
            T_corr = Spectrum.__correct(left.w, left.T, right.w, right.T, kind)
            right.R = R_corr['right_y_corr']
            right.T = T_corr['right_y_corr']
            self.FULL = RTPair(R_corr['stitched_x'], R_corr['stitched_y'], T_corr['stitched_y'])

    def stitched(self, num_points=1000):
        """
        Calculates stitched spectrum. Resulting spectrum is obtained using interpolation.

        Parameters
        ----------
        num_points : int
            Number of points in the resulting spectrum.

        Returns
        -------
        RTPair
            Stitched spectrum.
        """
        # Perform stitching for two-detector case.
        if len(self.detectors) == 2:
            # Check for correctness.
            if self.detectors == {'VIS', 'MIR'}:
                raise Exception('Cannot MIR using VIS --- NIR is needed.')
            if len(self.detectors) == 1:
                raise Exception('Only one spectrum is provided — nothing to correct.')
            # Alias those two spectra.
            if self.detectors == {'VIS', 'NIR'}:
                left, right = self.VIS, self.NIR
            else:
                left, right = self.NIR, self.MIR
            # Figure out transition region limits.
            w0, w1 = right.w[0], left.w[-1]
        else:
            raise Exception(f'Only 2-detector spectra could be stitched.')

    @property
    def min_wl(self):
        if self.VIS is not None:
            return self.VIS.w[0]
        elif self.NIR is not None:
            return self.NIR.w[0]
        elif self.MIR is not None:
            return self.MIR.w[0]
        else:
            raise Exception('No data provided!')

    @property
    def max_wl(self):
        if self.MIR is not None:
            return self.MIR.w[-1]
        elif self.NIR is not None:
            return self.NIR.w[-1]
        elif self.VIS is not None:
            return self.VIS.w[-1]
        else:
            raise Exception('No data provided!')
