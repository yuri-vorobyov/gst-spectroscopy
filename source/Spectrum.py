import numpy as np
import matplotlib.pyplot as plt
from sg_smooth.smoothing import smSG_bisquare


class Spectrum:
    """
    Spectrum is a collection of several individual spectra — obtained using different detectors at some particular
    sample temperature.
    """

    WAVELENGTH_MIN = {
        'Si': 630,  # nm
        'InGaAs': 850,  # nm
        'DTGS': 1350  # nm
    }
    WAVELENGTH_MAX = {
        'Si': 1160,  # nm
        'InGaAs': 2450,  # nm
        'DTGS': 2650  # nm (technically, this particular value is limited by the substrate, not the detector)
    }
    PRINT_HEADER = True
    COLORS = {
        'Si': '#1f77b5',
        'InGaAs': '#fd8114',
        'DTGS': '#d82a2d'
    }
    SMOOTHING_WINDOW_RADIUS = {
        'Si': 150,
        'InGaAs': 150,
        'DTGS': 150
    }
    SMOOTHING_POLY_ORDER = {
        'Si': 3,
        'InGaAs': 3,
        'DTGS': 3
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
        self.__raw_vis_data = None
        self.__raw_nir_data = None
        self.__raw_mir_data = None

        # Load raw data, convert it to the internal representation and save for later use
        if Si:
            data = np.loadtxt(Si, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self.__raw_vis_data = Spectrum.__strip_wl(data, 'Si')
        if InGaAs:
            data = np.loadtxt(InGaAs, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self.__raw_nir_data = Spectrum.__strip_wl(data, 'InGaAs')
        if DTGS:
            data = np.loadtxt(DTGS, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self.__raw_mir_data = Spectrum.__strip_wl(data, 'DTGS')

        # Save value of temperature
        self.temperature = temperature

        self.__calculate_smoothed()

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
        Spectrum.__plot(self.__raw_vis_data, self.__raw_nir_data, self.__raw_mir_data,
                        signal, title)

    def plot_smoothed(self, signal='Signal', title=''):
        """Show the plot of smoothed spectra data."""
        Spectrum.__plot(self.__smoothed_vis_data, self.__smoothed_nir_data, self.__smoothed_mir_data,
                        signal, title)

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

    def __calculate_smoothed(self):
        self.__smoothed_vis_data = None
        self.__smoothed_nir_data = None
        self.__smoothed_mir_data = None
        if self.__raw_vis_data is not None:
            self.__smoothed_vis_data = np.column_stack(smSG_bisquare(self.__raw_vis_data[:, 0],
                                                                     self.__raw_vis_data[:, 1],
                                                                     Spectrum.SMOOTHING_WINDOW_RADIUS['Si'],
                                                                     Spectrum.SMOOTHING_POLY_ORDER['Si'],
                                                                     extend=False))
        if self.__raw_nir_data is not None:
            self.__smoothed_nir_data = np.column_stack(smSG_bisquare(self.__raw_nir_data[:, 0],
                                                                     self.__raw_nir_data[:, 1],
                                                                     Spectrum.SMOOTHING_WINDOW_RADIUS['InGaAs'],
                                                                     Spectrum.SMOOTHING_POLY_ORDER['InGaAs'],
                                                                     extend=False))
        if self.__raw_mir_data is not None:
            self.__smoothed_mir_data = np.column_stack(smSG_bisquare(self.__raw_mir_data[:, 0],
                                                                     self.__raw_mir_data[:, 1],
                                                                     Spectrum.SMOOTHING_WINDOW_RADIUS['DTGS'],
                                                                     Spectrum.SMOOTHING_POLY_ORDER['DTGS'],
                                                                     extend=False))


if __name__ == '__main__':
    Spectrum.PRINT_HEADER = True
    s = Spectrum(InGaAs='../data/2024-04-16/1000nm/R/R_270c3(GST_1000nm)_VIS_InGaAs_CaF2.csv',
                 DTGS='../data/2024-04-16/1000nm/R/R_270c3_(GST_1000nm)_DTGS_MIR_CaF2.csv')
    s.plot_smoothed(signal='R', title='2024-04-16')
