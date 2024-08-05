import numpy as np
import os.path
from scipy.interpolate import interp1d
from sg_smooth.smoothing import smSG_bisquare
import matplotlib.pyplot as plt


class RTPair:
    """
    Container for a pair of R and T spectra measured in one experiment using the same detector.
    Both R and T are from 0 to 1, and wavelength scale is in nm.
    This container optionally handles information about the detector which was used for spectra acquisition.
    """

    # Each spectrum has its default color for built-in plots.
    COLORS = {
        'R': '#1f77b5',  # blue-ish
        'T': '#fd8114'   # red-ish
    }

    # Each detector has its own spectrum interval
    DETECTORS = {
        'Si': {
            'type': 'VIS',
            'limits': (600, 1085)
        },
        'Vertex-Si': {
            'type': 'VIS',
            'limits': (600, 890)
        },
        'CaF2 MCT': {
            'type': 'NIR',
            'limits': (950, 2500)  # limited by 1737F substrate
        },
        'InGaAs': {
            'type': 'NIR',
            'limits': (840, 2400)
        },
        'Vertex-InGaAs': {
            'type': 'NIR',
            'limits': (850, 2400)
        }
    }

    def __init__(self, w, R, T, detector=None):
        """
        Parameters
        ----------
        w : array-like
            Array of wavelengths in nanometers.
        R : array-like
            Reflectance spectra.
        T : array-like
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
        self.detector = detector
        self.w = w
        self.R = R
        self.T = T

        # Also in the form of single array (original data --- should remain untouched).
        self._data = np.column_stack((self.w, self.R, self.T))

        # Strip according to detector limits.
        if detector:
            self.strip(*RTPair.DETECTORS[detector]['limits'])

        # Placeholders for the smoothed version.
        self.sw = None
        self.sR = None
        self.sT = None

    @classmethod
    def from_ftir_files(cls, R, T, detector=None, same_scale=True):
        """
        Factory method to instantiate `RTPair` from text files with the spectra.

        Parameters
        ----------
        R : str
            File path to the R spectrum.
        T : str
            File path to the T spectrum.
        detector : str or None
            Detector type which was used for the spectra acquisition. Default is `None` meaning that no information
            about detector is available.
        same_scale : bool
            If `True`, the same scale is assumed for R and T spectra. If `False`, then the spectra will be resampled
            to the one of the smaller (smaller number of points) scale.
        """
        # Check file existence.
        if not os.path.exists(R):
            raise Exception(f'"{R}" cannot be found!')
        if not os.path.exists(T):
            raise Exception(f'"{T}" cannot be found!')
        # Load the data.
        r = np.loadtxt(R, skiprows=1, dtype=np.float64)
        if len(r.shape) != 2 or r.shape[1] != 2:
            raise Exception('R dataset must have 2 columns.')
        t = np.loadtxt(T, skiprows=1, dtype=np.float64)
        if len(t.shape) != 2 or t.shape[1] != 2:
            raise Exception('T dataset must have 2 columns.')
        # Ensure spectra are sorted.
        # todo : make it happen
        # Check scale of the spectra and resample if needed.
        if same_scale:
            # Check whether wave-number scales equal to each other.
            if len(r[:, 0]) == len(t[:, 0]):
                if not np.allclose(r[:, 0], t[:, 0], rtol=1e-6):
                    raise Exception(f'Looks like "{R}" and "{T}" are from different data sets --- wave-number scales'
                                    'are different. Try `same_scale=False`.')
            else:
                raise Exception(f'\n"{R}" : {len(r[:, 0])} points\n"{T}" : {len(t[:, 0])} points.'
                                'Try `same_scale=False`.')
            # Instantiate RTPair.
            return cls(1e7 / ((r[:, 0] + t[:, 0]) / 2), r[:, 1], t[:, 1], detector)
        else:
            # Find overlapping range.
            start_r, finish_r = float(r[0, 0]), float(r[-1, 0])
            start_t, finish_t = float(t[0, 0]), float(t[-1, 0])
            start_wn, finish_wn = max(start_r, start_t), min(finish_r, finish_t)
            # Strip both datasets so that their spectral scales are overlapping.
            r = r[(r[:, 0] > start_wn) * (r[:, 0] < finish_wn), :]
            t = t[(t[:, 0] > start_wn) * (t[:, 0] < finish_wn), :]
            # Resample larger spectrum.
            if len(r[:, 0]) > len(t[:, 0]):  # R is larger
                wn_scale = t[:, 0]
                R = interp1d(r[:, 0], r[:, 1], 'cubic', assume_sorted=True)(wn_scale)
                return cls(1e7 / wn_scale, R, t[:, 1], detector)
            else:  # T is larger or same
                wn_scale = r[:, 0]
                T = interp1d(t[:, 0], t[:, 1], 'cubic', assume_sorted=True)(wn_scale)
                return cls(1e7 / wn_scale, r[:, 1], T, detector)

    @property
    def e(self):
        """Return photon energy scale for this spectrum."""
        return 1239.842 / self.w

    @property
    def se(self):
        return 1239.842 / self.sw

    def strip(self, wl_min, wl_max):
        """
        Strip the wavelength scale. It is the raw input spectra which are getting stripped, so this method could be
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

    def strip_by_detector(self):
        """ Strip the wavelength scale using the specified detector limits. """
        if self.detector:
            self.strip(*RTPair.DETECTORS[self.detector]['limits'])
        else:
            raise Exception('Detector is not specified. Use `RTPair.strip()` instead.')

    def calc_smoothed(self, w, n):
        """
        Calculate (update) filtered spectra.

        Parameters
        ----------
        w : int
            Radius of smoothing window (number of points).
        n : int
            Order of approximating polynomial.
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
