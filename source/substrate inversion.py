"""
substrate inversion.py

Calculate n and k spectra of substrate sample from its corresponding R&T spectra using root-finding method.

Interference is neglected.
"""
from Spectrum import Spectrum, RTPair
from OpticalConstantsSpectrum import OpticalConstantsSpectrum as OCS
from calc import calc_RT_ASA
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
plt.rcParams['savefig.directory'] = '.'
COLORS = [item['color'] for item in plt.rcParams['axes.prop_cycle'].__dict__['_left']]


# From FTIR (Vertex, 02.04.2024)
meas = Spectrum(VIS_T='../test data/T_glass_Si.csv',
                VIS_R='../test data/R_glass_Si.csv',
                VIS_detector='Vertex-Si',
                NIR_T='../test data/T_glass_InGaAs.csv',
                NIR_R='../test data/R_glass_InGaAs.csv',
                NIR_detector='Vertex-InGaAs')
meas.calculate_corrected(kind='uniform')
# meas.plot(), quit()  # uncomment to check if everything is correct
rt = meas.FULL

# From spectrophotometry
# data = np.loadtxt('../test data/Corning Glass.csv', skiprows=2)
# rt = RTPair(data[:, 0], data[:, 2] / 100, data[:, 1] / 100)
# rt.strip(400, 2500)  # strip UV

# rt.plot()  # just to check if data was loaded correctly
wavelengths = rt.w

# Substrate thickness is known.
d_sub = 0.7e-3 * 1e9  # nm

# NR root-finding is used to obtain n, k pairs for each wavelength.
ns, ks = [], []  # containers for n and k values
x0 = [1.5, 0]  # initial guess [n, k]
for w, r_meas, t_meas in zip(wavelengths, rt.R, rt.T):
    print(f'solve at {w:.1f} nm, R_meas = {r_meas:.3f}, T_meas = {t_meas:.3f}')


    def f(x):
        """The function for root finding."""
        n, k = x
        r_calc, t_calc = calc_RT_ASA(w, n, k, d_sub)
        return r_calc - r_meas, t_calc - t_meas


    # Solution.
    res = root(f, np.asarray(x0))
    if res.success:
        print(f'    {res.x}')
        ns.append(res.x[0])
        ks.append(res.x[1])
        x0 = res.x
    else:
        raise Exception('Could not converge.')


ocs = OCS(wavelengths, ns, ks)
ocs.plot()

# Finally, save the n and k for latter use.
ocs.save('substrate (w,n,k).txt')
print('saved')
