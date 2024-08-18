import matplotlib.pyplot as plt
import numpy as np
import os.path


class OpticalConstantsSpectrum:

    # Each spectrum has its default color for built-in plots.
    COLORS = {
        'n': '#1f77b5',  # blue-ish
        'k': '#fd8114'   # red-ish
    }

    def __init__(self, w, n, k):
        # Check correctness of input arrays.
        if not (len(w) == len(n) == len(k)):
            raise Exception('Input arrays lengths must be equal.')

        # Save data.
        self.w = w  # in nm
        self.n = n
        self.k = k

        # Also in the form of single array (original data --- should remain untouched).
        self._data = np.column_stack((self.w, self.n, self.k))

    @classmethod
    def from_wnk_file(cls, fname):
        # Check file existence.
        if not os.path.exists(fname):
            raise Exception(f'"{fname}" cannot be found!')
        # Load the data.
        data = np.loadtxt(fname, skiprows=0, dtype=np.float64)
        if len(data.shape) != 2 or data.shape[1] != 3:
            raise Exception('Dataset must have 3 columns.')
        # Ensure spectra are sorted.
        # todo : make it happen
        # Instantiate RTPair.
        return cls(data[:, 0], data[:, 1], data[:, 2])

    @property
    def e(self):
        """Return photon energy scale for this spectrum in eV."""
        return 1239.842 / self.w

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
            l_n, = ax_n.plot(x, self.n, c=OpticalConstantsSpectrum.COLORS['n'], alpha=0.7, label='n')
            l_k, = ax_k.plot(x, self.k, c=OpticalConstantsSpectrum.COLORS['k'], alpha=0.7, label='k')
            ax_n.legend(handles=(l_n, l_k), loc='best')
        elif scale == 'nk':
            ax.set_title(title)
            ax.set_xlabel('n')
            ax.set_ylabel('k')
            ax.plot(self.n, self.k, '.', ms=4, mec='none', alpha=0.7)
        elif scale == 'Tauc':
            ax.set_title(title)
            ax.set_xlabel(r'Photon energy (eV)')
            ax.set_ylabel(r'$ \mathbf{\mathrm{\left(\alpha E\right)^{1/2}\,(cm^{-1})}} $')
            alpha = 4 * np.pi * self.k / self.w * 1e7
            ax.plot(self.e, (alpha * self.e)**0.5, '.', ms=4, c=OpticalConstantsSpectrum.COLORS['k'], alpha=0.7)

        plt.show(block=True)
