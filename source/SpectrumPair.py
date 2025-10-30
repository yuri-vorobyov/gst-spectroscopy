import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class SpectrumPair:
    """
    Container for a pair of spectra. Horizontal scale is wavelength, and it is stored in nanometers.
    Values of physical quantities are considered non-negative.
    Below individual spectra are called first and second.
    """

    # Each spectrum has its color for built-in plots.
    COLORS = {
        'first': '#1f77b5',  # blue-ish
        'second': '#fd8114'  # red-ish
    }

    def __init__(self, w, first, second, first_label, second_label):
        """
        Parameters
        ----------
        w : array-like
            Array of wavelengths in nanometers.
        first : array-like
            First spectrum.
        second : array-like
            Second spectrum.
        first_label : str
            Label of the first spectrum 
        second_label : str
            Label of the second spectrum 
        """
        # Check correctness of input arrays.
        if not (len(w) == len(first) == len(second)):
            raise Exception('Input arrays lengths must be equal.')

        # Save labels.
        self.first_label = first_label
        self.second_label = second_label

        # Ensure spectra are sorted by the wavelength.
        ii = np.argsort(w)
        self.w = np.asarray(w)[ii]
        self.first = np.asarray(first)[ii]
        self.second = np.asarray(second)[ii]

        # Remove negative values.
        ii = (self.first > 0) * (self.second > 0)
        self.w = self.w[ii]
        self.first = self.first[ii]
        self.second = self.second[ii]

    @property
    def e(self):
        """Return photon energy scale in eV for this spectrum."""
        return 1239.842 / self.w

    @property
    def sE(self):
        return 1239.842 / self.sw

    def strip(self, wl_min, wl_max):
        """
        Strip the wavelength scale.

        Parameters
        ----------
        wl_min : float
            Minimum wavelength in nm.
        wl_max : float
            Maximum wavelength in nm.
        """
        ii = (self.w > wl_min) * (self.w < wl_max)
        self.w = self.w[ii]
        self.first = self.first[ii]
        self.second = self.second[ii]

    def resample(self, *, step=None, scale=None):
        """
        Resample the spectrum using interpolation.
        """
        if scale is None:
            if step is None:
                # The step for resampling could be inferred from the data.
                step = (self.w[1:] - self.w[:-1]).max()
            scale = np.linspace(self.w[0], self.w[-1], int(np.ceil((self.w[-1] - self.w[0]) / step)) + 1)

        self.first = interp1d(self.w, self.first, kind='linear')(scale)
        self.second = interp1d(self.w, self.second, kind='linear')(scale)
        self.w = scale

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

        # Here we assume that "style.mplstyle" is placed near this script.
        plt.style.use(Path(__file__).parent / 'style.mplstyle')
        plt.rcParams['savefig.directory'] = Path(__file__).parent
        fig, ax_first = plt.subplots(1, 1)
        fig.canvas.manager.set_window_title(title)
        ax_second = ax_first.twinx()
        ax_first.set_title(title)
        ax_first.set_xlabel({'wavelength': 'Wavelength (nm)',
                             'energy': 'Photon energy (eV)'}[scale])
        ax_first.set_ylabel(self.first_label)
        ax_second.set_ylabel(self.second_label)

        x = {'wavelength': self.w, 'energy': self.e}[scale]
        y1 = self.first
        y2 = self.second

        l_first, = ax_first.plot(x, y1, c=SpectrumPair.COLORS['first'], alpha=0.7, label=self.first_label)
        l_second, = ax_second.plot(x, y2, c=SpectrumPair.COLORS['second'], alpha=0.7, label=self.second_label)

        ax_second.legend(handles=(l_first, l_second), loc='best')  # because "second" is above the "first"

        plt.show(block=True)
