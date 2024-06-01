"""
substrate inversion.py

Calculate n and k spectra of 1737F from its corresponding R&T spectra using root-finding method.

Spectrum of n is fitted by Sellmeier equation.
"""
from Spectrum import Spectrum
import tmm
from calc import calc_RT_ASA
import numpy as np
from scipy.optimize import root, curve_fit, root_scalar
from optical_constants.optical_constants import Sellmeier
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
plt.rcParams['savefig.directory'] = '.'
COLORS = [item['color'] for item in plt.rcParams['axes.prop_cycle'].__dict__['_left']]


# Load substrate spectra.
meas = Spectrum(VIS_T='../data/2024-04-02/1737F/T/T_glass_Si.csv',
                VIS_R='../data/2024-04-02/1737F/R/R_glass_Si.csv',
                NIR_T='../data/2024-04-02/1737F/T/T_glass_InGaAs.csv',
                NIR_R='../data/2024-04-02/1737F/R/R_glass_InGaAs.csv',
                MIR_T='../data/2024-04-02/1737F/T/T_glass_DTGS_MIR.csv',
                MIR_R='../data/2024-04-02/1737F/R/R_glass_DTGS_MIR.csv')

# Get the wavelength scale and the energy scale.
rt = meas.NIR
wavelengths = rt.w
photon_energies = rt.e

# Thickness is known.
d_sub = 0.7e-3 * 1e9  # nm

# NR root-finding is used to obtain n, k pairs for each wavelength.
ns, ks = [], []
for w, r_meas, t_meas in zip(wavelengths, rt.R, rt.T):
    print(f'solve at {w:.1f} nm, R_meas = {r_meas:.3f}, T_meas = {t_meas:.3f}')


    def f(x):
        """The function for root finding."""
        n, k = x
        r_calc, t_calc = calc_RT_ASA(w, n, k, d_sub)
        return r_calc - r_meas, t_calc - t_meas


    # Solution.
    x0 = [1.5, 0]
    res = root(f, np.asarray(x0))
    if res.success:
        print(f'    {res.x}')
        ns.append(res.x[0])
        ks.append(res.x[1])
    else:
        raise Exception('Could not converge.')


# Next, to avoid introducing extra noise down the line, we approximate n by Sellmeier equation.
def f(wl, B1, B2, B3, C1, C2, C3):
    dispersion = Sellmeier(B1, B2, B3, C1, C2, C3)
    return dispersion.n(1239.842 / wl)


p0 = [2.9e-1, 9.8e-1, 3.7e+1, 1.0e-2, 1.0e-2, 6.4e+3]
popt, pcov = curve_fit(f, wavelengths, ns, p0 )
disp = Sellmeier(*popt)


# Create figure.
fig, ax_n = plt.subplots(1, 1, constrained_layout=True)
ax_k = ax_n.twinx()
fig.canvas.manager.set_window_title('figure')

# Configure axes.
ax_n.set_xlabel(r'Wavelength (nm)')
ax_n.set_ylabel(r'n')
ax_k.set_ylabel(r'k')

# Plot n and k.
l_n, = ax_n.plot(wavelengths, ns, c=COLORS[0], label='n')
ax_n.plot(wavelengths, f(wavelengths, *popt), c=COLORS[2])
l_k, = ax_k.plot(wavelengths, ks, c=COLORS[1], label='k')

# Create legend.
ax_n.legend(handles=[l_n, l_k], loc='upper center')

# Show the title.
ax_n.set_title('1737F')

# Show the window.
plt.show()

# Finally, save the n and k for latter use.
np.savetxt('1737F.txt', np.column_stack((wavelengths, ns, ks)))