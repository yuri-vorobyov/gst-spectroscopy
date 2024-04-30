import numpy as np
import matplotlib.pyplot as plt


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

        # Load raw data, convert it to the internal representation and save for later use
        if Si:
            data = np.loadtxt(Si, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self.raw_vis_data = Spectrum.__strip_wl(data, 'Si')
        if InGaAs:
            data = np.loadtxt(InGaAs, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self.raw_nir_data = Spectrum.__strip_wl(data, 'InGaAs')
        if DTGS:
            data = np.loadtxt(DTGS, skiprows=1, dtype=np.float64)
            data = Spectrum.__convert_to_nm(data)
            self.raw_mir_data = Spectrum.__strip_wl(data, 'DTGS')

        # Save value of temperature
        self.temperature = temperature

    def plot_raw_spectra(self, spectra):
        """Show the plot of raw spectra data."""
        if 'VIS' in spectra:
            plt.plot(self.raw_vis_data[:, 0], self.raw_vis_data[:, 1])
        if 'NIR' in spectra:
            plt.plot(self.raw_nir_data[:, 0], self.raw_nir_data[:, 1])
        if 'MIR' in spectra:
            plt.plot(self.raw_mir_data[:, 0], self.raw_mir_data[:, 1])

        plt.show()

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


if __name__ == '__main__':
    Spectrum.PRINT_HEADER = True
    s = Spectrum(InGaAs='../data/2024-04-16/1000nm/R/R_270c3(GST_1000nm)_VIS_InGaAs_CaF2.csv',
                 DTGS='../data/2024-04-16/1000nm/R/R_270c3_(GST_1000nm)_DTGS_MIR_CaF2.csv')
