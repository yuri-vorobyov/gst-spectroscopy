import numpy as np
import matplotlib.pyplot as plt
from sg_smooth.smoothing import smSG_bisquare
from scipy.interpolate import interp1d
import os.path


class RTPair:
    """
    Container for a pair of R and T spectra measured in one experiment using same detector.
    """


    def __init__(self, R, T):
        """
        Parameters
        ----------
        R : str
            File path to the R spectrum.
        T : str
            File path to the T spectrum.
        """
        # Check file existence.
        if not os.path.exists(R):
            raise Exception(f'"{R}" cannot be found!')
        if not os.path.exists(T):
            raise Exception(f'"{T}" cannot be found!')
        # Load the data.
        r = np.loadtxt(R, skiprows=1, dtype=np.float64)
        t = np.loadtxt(T, skiprows=1, dtype=np.float64)
        # Check is wavelength scale is the same.
        if not np.allclose(r[:, 0], t[:, 0], rtol=1e-6):
            raise Exception('Looks like R and T are from different data sets --- wavelength scales are different.')
        # Convert wavelength scale to nm and save for latter use.
        self.w = 1e7 / r[:, 0]
        self.R = r[:, 1]
        self.T = t[:, 1]
        # Also in the form of single array (should remain untouched).
        self._data = np.column_stack((self.w, self.R, self.T))

    @property
    def e(self):
        return 1239.842 / self.w

    def strip(self, wl_min, wl_max):
        """
        Strip the wavelength scale. It is the initial spectra which are getting stripped, so this method could be
        invoked several times and will provide correct results.

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


class Spectrum:
    """
    Spectrum is a collection of several individual spectra — obtained using different detectors at some particular
    sample temperature.

    Water vapor has several absorption bands, according to literature: at 1.38, 1.87, 2.7, 6.3 um. In out experimental
    spectra two of them contribute significantly:
        1340 – 1495 nm
        1785 – 1970 nm
    """

    WAVELENGTH_MIN = {
        'Si': 630,  # nm
        'InGaAs': 850,  # nm
        'DTGS': 1350  # nm
    }
    WAVELENGTH_MAX = {
        'Si': 1160,  # nm
        'InGaAs': 2450,  # nm
        'DTGS': 6650  # nm
    }
    PRINT_HEADER = False
    COLORS = {
        'Si': '#1f77b5',
        'InGaAs': '#fd8114',
        'DTGS': '#d82a2d'
    }
    SMOOTHING_WINDOW_RADIUS = {
        'Si': 20,
        'InGaAs': 30,
        'DTGS': 150
    }
    SMOOTHING_POLY_ORDER = {
        'Si': 3,
        'InGaAs': 3,
        'DTGS': 3
    }
    WATER_BANDS = {
        'A': (1340, 1495),
        'B': (1785, 1970)
    }

    def __init__(self, Si='', InGaAs='', DTGS='', temperature='RT'):
        """
        Create a Spectrum instance, composed of up to three spectra obtained using different detectors.

        :param Si: Filename for the data from Si (VIS) detector.
        :param InGaAs: Filename for the data from InGaAs (NIR) detector.
        :param DTGS: Filename for the data from DTGS (MIR) detector.
        :param temperature: Value of temperature (in K) at which the spectra were collected, or 'RT'.
        """
        # If necessary to do so, print headers to make sure there is no error in locating files.
        if Spectrum.PRINT_HEADER:
            if Si:
                print(f'VIS: {Spectrum.__file_header(Si)}')
            if InGaAs:
                print(f'NIR: {Spectrum.__file_header(InGaAs)}')
            if DTGS:
                print(f'MIR: {Spectrum.__file_header(DTGS)}')

        # Initialize instance variables.
        self._raw_vis_data = None
        self._raw_nir_data = None
        self._raw_mir_data = None
        self._smoothed_vis_data = None
        self._smoothed_nir_data = None
        self._smoothed_mir_data = None
        self.__corrected_nir_data = None
        self.__corrected_mir_data = None
        self.detectors = set()

        # Load raw data, convert it to the internal representation and save for later use
        if Si:
            self.detectors.add('VIS')
            data = np.loadtxt(Si, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self._raw_vis_data = Spectrum.__strip_wl(data, 'Si')
        if InGaAs:
            self.detectors.add('NIR')
            data = np.loadtxt(InGaAs, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self._raw_nir_data = Spectrum.__strip_wl(data, 'InGaAs')
        if DTGS:
            self.detectors.add('MIR')
            data = np.loadtxt(DTGS, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self._raw_mir_data = Spectrum.__strip_wl(data, 'DTGS')

        # Save value of temperature
        self.temperature = temperature

        # self.__calculate_smoothed()
        # self.__calculate_corrected(kind='linear')

    @staticmethod
    def __assert_detector(detector):
        if detector not in {'VIS', 'NIR', 'MIR'}:
            raise Exception('`detector` should be one of "VIS", "NIR", or "MIR"')

    def __assert_detector_in_use(self, detector):
        if detector not in self.detectors:
            raise Exception(f'No data provided for {detector}, only {self.detectors} are given.')

    @staticmethod
    def __plot(data_vis, data_nir, data_mir, signal='Signal', title=''):
        """
        Plot the spectrum.

        :param data_vis: A VIS spectrum in the table form (1st column for wavelength in nm, 2nd — for signal intensity).
        :param data_nir: A NIR spectrum in the table form (1st column for wavelength in nm, 2nd — for signal intensity).
        :param data_mir: A MIR spectrum in the table form (1st column for wavelength in nm, 2nd — for signal intensity).
        :param signal: Label of the vertical axis.
        :param title: Plot title.
        """
        plt.style.use('style.mplstyle')
        plt.rcParams['savefig.directory'] = '.'
        fig, ax = plt.subplots(1, 1)
        fig.canvas.manager.set_window_title(title)
        ax.set_title(title)
        ax.set_xlabel(r'Wavelength (nm)')
        ax.set_ylabel(signal)

        if data_vis is not None:
            ax.plot(data_vis[:, 0], data_vis[:, 1], c=Spectrum.COLORS['Si'], alpha=0.7)
        if data_nir is not None:
            ax.plot(data_nir[:, 0], data_nir[:, 1], c=Spectrum.COLORS['InGaAs'], alpha=0.7)
        if data_mir is not None:
            ax.plot(data_mir[:, 0], data_mir[:, 1], c=Spectrum.COLORS['DTGS'], alpha=0.7)

        plt.show(block=True)

    def plot_raw(self, signal='Signal', title=''):
        """Show the plot of raw spectra data."""
        Spectrum.__plot(self._raw_vis_data, self._raw_nir_data, self._raw_mir_data,
                        signal, title)

    def plot_smoothed(self, signal='Signal', title=''):
        """Show the plot of smoothed spectra data."""
        Spectrum.__plot(self._smoothed_vis_data, self._smoothed_nir_data, self._smoothed_mir_data,
                        signal, title)

    def plot_corrected(self, signal='Signal', title=''):
        """Show the plot of corrected spectra data."""
        vis, nir, mir = None, None, None
        if 'VIS' in self.detectors:
            vis = self._raw_vis_data
        if 'NIR' in self.detectors:
            if self.__corrected_nir_data is not None:
                nir = self.__corrected_nir_data
            else:
                nir = self._raw_nir_data
        if 'MIR' in self.detectors:
            if self.__corrected_mir_data is not None:
                mir = self.__corrected_mir_data
            else:
                mir = self._raw_mir_data

        Spectrum.__plot(vis, nir, mir, signal, title)

    @staticmethod
    def __file_header(fname):
        """Returns first line of a text file (hopefully with spectrum info)."""
        with open(fname, mode='r', encoding='utf8') as fs:
            return fs.readline().rstrip()

    @staticmethod
    def __convert_to_nm(data):
        """
        Converts the input spectrum from wave-numbers scale (in cm^-1) to wavelength scale (in nm).

        :param data: Spectrum soon to be converted.
        :return: Converted spectrum.
        """
        data[:, 0] = 1e7 / data[:, 0]  # convert from cm^-1 to nm
        return data

    @staticmethod
    def __strip_wl(data, detector):
        """
        Returns the part of the input spectrum residing in between the detector limits.

        :param data: Spectrum to be stripped.
        :param detector: 'Si', or 'InGaAs', or 'DTGS'.
        :return: Stripped spectrum
        """
        min_wl, max_wl = Spectrum.WAVELENGTH_MIN[detector], Spectrum.WAVELENGTH_MAX[detector]
        data = data[(data[:, 0] > min_wl) * (data[:, 0] < max_wl)]
        return data

    def _calculate_smoothed(self):
        if self._raw_vis_data is not None:
            self._smoothed_vis_data = np.column_stack(smSG_bisquare(self._raw_vis_data[:, 0],
                                                                    self._raw_vis_data[:, 1],
                                                                    Spectrum.SMOOTHING_WINDOW_RADIUS['Si'],
                                                                    Spectrum.SMOOTHING_POLY_ORDER['Si'],
                                                                    extend=False))
        if self._raw_nir_data is not None:
            self._smoothed_nir_data = np.column_stack(smSG_bisquare(self._raw_nir_data[:, 0],
                                                                    self._raw_nir_data[:, 1],
                                                                    Spectrum.SMOOTHING_WINDOW_RADIUS['InGaAs'],
                                                                    Spectrum.SMOOTHING_POLY_ORDER['InGaAs'],
                                                                    extend=False))
        if self._raw_mir_data is not None:
            self._smoothed_mir_data = np.column_stack(smSG_bisquare(self._raw_mir_data[:, 0],
                                                                     self._raw_mir_data[:, 1],
                                                                    Spectrum.SMOOTHING_WINDOW_RADIUS['DTGS'],
                                                                    Spectrum.SMOOTHING_POLY_ORDER['DTGS'],
                                                                    extend=False))

    def _calculate_corrected(self, kind='uniform'):
        # Check for correctness.
        if self.detectors == {'VIS', 'MIR'}:
            raise Exception('Cannot stitch VIS with MIR.')
        if len(self.detectors) == 1:
            raise Exception('Only one spectrum is provided — nothing to stitch.')

        # Perform stitching for two-detector case.
        if len(self.detectors) == 2:
            # Alias those two spectra.
            if self.detectors == {'VIS', 'NIR'}:
                left, right = self._raw_vis_data, np.copy(self._raw_nir_data)
                left_sm, right_sm = self._smoothed_vis_data, self._smoothed_nir_data
            else:
                left, right = self._raw_nir_data, np.copy(self._raw_mir_data)
                left_sm, right_sm = self._smoothed_nir_data, self._smoothed_mir_data

            # Interpolate for the same set of wavelengths.
            w0, w1 = right_sm[0, 0], left_sm[-1, 0]
            N = 500
            w = np.linspace(w0 + 1, w1 - 1, N)
            f = interp1d(left_sm[left_sm[:, 0] >= w0, 0],
                         left_sm[left_sm[:, 0] >= w0, 1],
                         'cubic', assume_sorted=True)(w)
            g = interp1d(right_sm[right_sm[:, 0] <= w1, 0],
                         right_sm[right_sm[:, 0] <= w1, 1],
                         'cubic', assume_sorted=True)(w)

            # Correct right spectrum.
            A = (g * g).sum()
            D = (f * g).sum()
            if kind == 'uniform':
                b = D / A
                right[:, 1] = right[:, 1] * b
            else:
                B = (w * g * g).sum()
                C = (w * w * g * g).sum()
                E = (w * f * g).sum()
                a0 = (C * D - E * B) / (C * A - B * B)
                a1 = (E * A - D * B) / (C * A - B * B)
                right[:, 1] = (a0 + a1 * right[:, 0]) * right[:, 1]

            # And save it.
            if self.detectors == {'VIS', 'NIR'}:
                self.__corrected_nir_data = right
            else:
                self.__corrected_mir_data = right

    @property
    def min_wl(self):
        if 'VIS' in self.detectors:
            return self._raw_vis_data[0, 0]
        elif 'NIR' in self.detectors:
            return self._raw_nir_data[0, 0]
        else:
            return self._raw_mir_data[0, 0]

    @property
    def max_wl(self):
        if 'MIR' in self.detectors:
            return self._raw_mir_data[-1, 0]
        elif 'NIR' in self.detectors:
            return self._raw_nir_data[-1, 0]
        else:
            return self._raw_vis_data[-1, 0]

    def get_wavelengths(self, detector):
        """
        Returns the wavelength vector for the specified detector.

        Parameters
        ----------
        detector : str
            Data corresponding to which detector should be returned.

        Returns
        -------
        out : ndarray
            Wavelength vector.
        """
        Spectrum.__assert_detector(detector)
        self.__assert_detector_in_use(detector)
        if detector == 'VIS':
            return self._raw_vis_data[:, 0]
        elif detector == 'NIR':
            return self._raw_nir_data[:, 0]
        elif detector == 'MIR':
            return self._raw_mir_data[:, 0]

    def interpolate(self, data_kind='raw'):
        """
        Returns the interpolation function (result of interp1d).

        :param data_kind: one of 'raw', 'smoothed'
        """
        # @todo currently only single-detector spectra are supported
        if len(self.detectors) != 1:
            raise Exception('Only single-detector spectra are supported currently.')

        choose = {'VIS': {'raw': self._raw_vis_data, 'smoothed': self._smoothed_vis_data},
                  'NIR': {'raw': self._raw_nir_data, 'smoothed': self._smoothed_nir_data},
                  'MIR': {'raw': self._raw_mir_data, 'smoothed': self._smoothed_mir_data}}

        d = choose[next(iter(self.detectors))][data_kind]
        x, y = d[:, 0], d[:, 1]

        return interp1d(x, y, kind='cubic')


if __name__ == '__main__':
    s = Spectrum(InGaAs='../data/2024-04-16/1000nm/R/R_270c3(GST_1000nm)_VIS_InGaAs_CaF2.csv',
                 DTGS='../data/2024-04-16/1000nm/R/R_270c3_(GST_1000nm)_DTGS_MIR_CaF2.csv')
    s.plot_corrected(signal='R', title='2024-04-16')
