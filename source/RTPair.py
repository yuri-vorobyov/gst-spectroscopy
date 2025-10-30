import numpy as np
import os.path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from SpectrumPair import SpectrumPair


class RTPair(SpectrumPair):
    """
    Container for a pair of R and T spectra.
    Both R and T are non negative, and wavelength scale is in nm.
    This container optionally handles information about the detector which was used for spectra acquisition.
    """

    # Each spectrum has its default color for built-in plots.
    COLORS = {
        'R': '#1f77b5',  # blue-ish
        'T': '#fd8114'  # red-ish
    }

    # Each detector has its own spectrum interval
    DETECTORS = {
        'Hyperion Si': {
            'type': 'VIS',
            'limits': (600, 1030)
        },
        'Vertex Si': {
            'type': 'VIS',
            'limits': (600, 890)
        },
        'Hyperion MCT': {
            'type': 'NIR',
            'limits': (970, 2500)  # limited by 1737F substrate
        },
        'InGaAs': {
            'type': 'NIR',
            'limits': (840, 2400)
        },
        'Vertex InGaAs': {
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
        # Instantiate superclass. It will perform all the checks and sanitizing and will raise exception if needed.
        super().__init__(w, R, T, 'R', 'T')

        # Check if the detector type provided is supported. Information about the detector is optional.
        if detector:
            if detector not in RTPair.DETECTORS.keys():
                raise Exception(f'"{detector}" detector is not supported. Add info to the `RTPair.DETECTORS` or use `None` instead.')

        # Save data.
        self.detector = detector

        # Strip according to detector limits, if detector info is provided.
        if detector:
            self.strip(*RTPair.DETECTORS[detector]['limits'])

    @property
    def R(self):
        return self.first

    @property
    def T(self):
        return self.second

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
            raise Exception(f'"{os.path.abspath(R)}" cannot be found!')
        if not os.path.exists(T):
            raise Exception(f'"{os.path.abspath(T)}" cannot be found!')

        # Load the data.
        try:
            r = np.loadtxt(R, skiprows=1, dtype=np.float64, delimiter=',')
        except:
            raise Exception(f'Error while opening a file {os.path.abspath(R)}')
        if len(r.shape) != 2 or r.shape[1] != 2:
            raise Exception('R dataset must have 2 columns.')
        try:
            t = np.loadtxt(T, skiprows=1, dtype=np.float64, delimiter=',')
        except:
            raise Exception(f'Error while opening a file {os.path.abspath(T)}')
        if len(t.shape) != 2 or t.shape[1] != 2:
            raise Exception('T dataset must have 2 columns.')

        # Check scale of the spectra and resample if needed.
        if same_scale:
            # Check whether wave-number scales equal to each other.
            if len(r[:, 0]) == len(t[:, 0]):
                if not np.allclose(r[:, 0], t[:, 0], rtol=1e-6):
                    raise Exception(f'Looks like "{R}" and "{T}" are from different data sets --- wave-number scales'
                                    'are different. Try `same_scale=False` to resample.')
            else:
                raise Exception(f'\n"{R}" : {len(r[:, 0])} points\n"{T}" : {len(t[:, 0])} points.'
                                'Try `same_scale=False`.')
            # Instantiate RTPair. As `r` and `t` are already considered equal, any of the m could be used.
            return cls(1e7 / r[:, 0], r[:, 1], t[:, 1], detector)
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

    def strip_by_detector(self):
        """ Strip the wavelength scale using the specified detector limits. """
        if self.detector:
            self.strip(*RTPair.DETECTORS[self.detector]['limits'])
        else:
            raise Exception('Detector is not specified. Use `RTPair.strip()` instead.')
