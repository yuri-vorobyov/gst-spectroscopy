"""
substrate inversion.py

Calculate n and k spectra of substrate sample from its corresponding R&T spectra using root-finding method.

Interference is neglected.
"""
from Spectrum import Spectrum, RTPair
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
                VIS_detector='Si',
                NIR_T='../test data/T_glass_InGaAs.csv',
                NIR_R='../test data/R_glass_InGaAs.csv',
                NIR_detector='InGaAs')
meas.calculate_corrected(kind='uniform')
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


# Create figure.
fig, ax_n = plt.subplots(1, 1, constrained_layout=True)
ax_k = ax_n.twinx()
fig.canvas.manager.set_window_title('figure')

# Configure axes.
ax_n.set_xlabel(r'Wavelength (nm)')
ax_n.set_ylabel(r'n')
ax_k.set_ylabel(r'k')

# Plot n and k.
kwargs = dict(lw=1.4, alpha=0.65)
l_n, = ax_n.plot(wavelengths, ns, c=COLORS[0], label='n', **kwargs)
l_k, = ax_k.plot(wavelengths, ks, c=COLORS[1], label='k', **kwargs)

# Create legend.
ax_n.legend(handles=[l_n, l_k], loc='upper center')

# Show the title.
ax_n.set_title('1737F')

# Show the window.
plt.show()

# Finally, save the n and k for latter use.
np.savetxt('substrate (w,n,k).txt', np.column_stack((wavelengths, ns, ks)))
print('saved')
