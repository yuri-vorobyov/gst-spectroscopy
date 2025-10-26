import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from SpectrumPair import SpectrumPair


class OpticalConstantsSpectrum(SpectrumPair):

    def __init__(self, w, n, k):
        super().__init__(w, n, k, 'n', 'k')

    @classmethod
    def from_wnk_file(cls, fname):
        # Check file existence.
        if not Path(fname).is_file():
            raise Exception(f'"{fname}" cannot be found!')

        # Load the data.
        data = np.loadtxt(fname, skiprows=0, dtype=np.float64)
        if len(data.shape) != 2 or data.shape[1] != 3:
            raise Exception('Dataset must have 3 columns.')

        # Instantiation.
        return cls(data[:, 0], data[:, 1], data[:, 2])

    @property
    def n(self):
        return self.first

    @property
    def k(self):
        return self.second

    @property
    def alpha(self):
        """Return absorption spectrum in m^-1."""
        return 4 * np.pi * self.k / self.w

    def plot(self, scale='wavelength', title=''):
        """
        Plot the spectrum.

        Parameters
        ----------
        scale : str
            One of `wavelength`, `energy`, `nk`, 'Tauc'.
        title : str
            Optional title.
        """
        if scale not in {'wavelength', 'energy', 'nk', 'Tauc'}:
            raise Exception('`scale` supports only "wavelength", "energy", "nk", or "Tauc"')

        plt.style.use('style.mplstyle')
        plt.rcParams['savefig.directory'] = '.'
        fig, ax = plt.subplots(1, 1)
        fig.canvas.manager.set_window_title(title)

        if scale in {'wavelength', 'energy'}:
            ax_n = ax
            ax_k = ax_n.twinx()
            ax_n.set_title(title)
            ax_n.set_xlabel({'wavelength': 'Wavelength (nm)',
                             'energy': 'Photon energy (eV)'}[scale])
            ax_n.set_ylabel('n')
            ax_k.set_ylabel('k')
            x = {'wavelength': self.w, 'energy': self.e}[scale]
            l_n, = ax_n.plot(x, self.n, c=SpectrumPair.COLORS['first'], label='n')
            l_k, = ax_k.plot(x, self.k, c=SpectrumPair.COLORS['second'], label='k')
            ax_k.legend(handles=(l_n, l_k), loc='best')
        elif scale == 'nk':
            ax.set_title(title)
            ax.set_xlabel('n')
            ax.set_ylabel('k')
            ax.plot(self.n, self.k, '.', ms=4, mec='none', alpha=0.7)
        elif scale == 'Tauc':
            ax.set_title(title)
            ax.set_xlabel(r'Photon energy (eV)')
            ax.set_ylabel(r'$ \mathbf{\mathrm{\left(\alpha E\right)^{1/2}\,(cm^{-1})}} $')
            kwargs = dict(ms=4, c=OpticalConstantsSpectrum.COLORS['k'], alpha=0.7)
            ax.plot(self.energy, (self.alpha * self.energy)**0.5, '.', **kwargs)

        return ax

    def show(self):
        plt.show()

    def save(self, fname):
        """
        Save the optical constants spectrum to a wnk text file.

        Parameters
        ----------
        fname : str
            File name.
        """
        np.savetxt(fname, np.column_stack((self.w, self.n, self.k)))
