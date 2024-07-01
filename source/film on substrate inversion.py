"""
film on substrate inversion.py

Calculate n and k spectra of a thin film from R&T spectra provided the substrate n and k are known.

Root-finding method is used.
"""
from Spectrum import Spectrum, RTPair
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import contourpy
import shapely
import itertools
from calc import calc_RT_AFSA

plt.style.use('style.mplstyle')
plt.rcParams['savefig.directory'] = '.'
COLORS = [item['color'] for item in plt.rcParams['axes.prop_cycle'].__dict__['_left']]

np.set_printoptions(precision=4)

# From FTIR (Vertex, 02.04.2024)
meas = Spectrum(VIS_T='../data/2024-04-02/130nm/T_254c(GST225_130nm)_Si.csv',
                VIS_R='../data/2024-04-02/130nm/R_254c(GST225_130nm)_Si.csv',
                VIS_detector='Si',
                NIR_T='../data/2024-04-02/130nm/T_254c(GST225_130nm)_InGaAs.csv',
                NIR_R='../data/2024-04-02/130nm/R_254c(GST225_130nm)_InGaAs.csv',
                NIR_detector='InGaAs')
meas.calculate_corrected(kind='uniform')
rt = meas.FULL

# From spectrophotometry.
# data = np.loadtxt('../data/Spectrophotometry/GST225.csv', skiprows=2)
# rt = RTPair(data[:, 0], data[:, 2] / 100, data[:, 1] / 100)
# rt.strip(400, 2500)  # use same range as for the substrate calculation

# rt.plot()  # just to check if the thin-film sample data was loaded correctly
wavelengths = rt.w

sub = np.loadtxt('substrate (w,n,k).txt')

# Check that wavelength scale is the same for the thin-film sample and the substrate data.
if not np.allclose(rt.w, sub[:, 0], rtol=1e-6):
    raise Exception('Wavelength scales are different!')
print(f'{len(rt.w)} data points are loaded')

# Thicknesses are known.
d_film = 131.6  # nm
d_sub = 0.7e-3 * 1e9  # nm

# Graphical method is implemented as follows. For each trial pair of values of n and k within the limits both
# T_meas - T_calc and R_meas - R_calc. Therefore, here we need the scale of possible n and k values.
lim_n, lim_k = [-0.2, 6], [-0.2, 3]
N_n, N_k = 30, 15
n_trial, k_trial = np.meshgrid(np.linspace(lim_n[0], lim_n[1], N_n),
                               np.linspace(lim_k[0], lim_k[1], N_k),
                               indexing='ij')

# Prepare containers for calculated values.
T_trial = np.empty((N_n, N_k))
R_trial = np.empty((N_n, N_k))


def calc_T_and_R(film_n, film_k, substrate_n, substrate_k, wavelength_nm):
    """Calculate T and R (as a tuple) of a film on a substrate."""
    r_calc, t_calc = calc_RT_AFSA(wavelength_nm, film_n, film_k, d_film, substrate_n, substrate_k, d_sub)
    return t_calc, r_calc


def update_trial_matrix(wavelength, sub_n, sub_k, measured_R, measured_T):
    T_arr, R_arr = calc_T_and_R(n_trial, k_trial, sub_n, sub_k, wavelength)
    for i in range(N_n):
        for j in range(N_k):
            T_trial[i, j], R_trial[i, j] = T_arr[i, j] - measured_T, R_arr[i, j] - measured_R


# Calculate the solution contours. Effectively, `z` is passed by reference so change to corresponding array will
# propagate into the contour generator object automatically.
cg_R = contourpy.contour_generator(n_trial, k_trial, R_trial, name='serial', line_type='Separate')
cg_T = contourpy.contour_generator(n_trial, k_trial, T_trial, name='serial', line_type='Separate')

# Check graphically (uncomment to show those plots with intersecting contours).
# index = 250
# print(f'{sub[index, 0]} nm')
# update_trial_matrix(sub[index, 0], sub[index, 1], sub[index, 2], rt.R[index], rt.T[index])
# plt.contour(n_trial, k_trial, R_trial, levels=[0], colors=COLORS[0])
# plt.contour(n_trial, k_trial, T_trial, levels=[0], colors=COLORS[1])
# plt.xlabel('n')
# plt.ylabel('k')
# plt.show()
# quit()

all_roots = []
# For each wavelength.
for index, wl in enumerate(wavelengths):
    n_sub, k_sub = sub[index, 1], sub[index, 2]
    t_meas, r_meas = rt.T[index], rt.R[index]
    print(f'{index:>4} solving for {wl:.1f} nm, n_sub = {n_sub:.3f}, k_sub = {k_sub:.3g}')

    update_trial_matrix(wl, n_sub, k_sub, r_meas, t_meas)

    roots_R = cg_R.lines(0.0)  # list of contour parts
    roots_T = cg_T.lines(0.0)  # list of contour parts

    # Find all intersection points, which are estimates of the solutions.
    roots = []
    for lsR, lsT in map(lambda x: map(shapely.LineString, x), itertools.product(roots_R, roots_T)):
        if lsR.intersects(lsT):
            intersection = shapely.intersection(lsR, lsT)
            if intersection.geom_type == 'MultiPoint':
                roots.extend(map(lambda x: (x.x, x.y), intersection.geoms))
            elif intersection.geom_type == 'Point':
                roots.append((intersection.x, intersection.y))
            else:
                raise Exception(f'Dont know what to do with {intersection.geom_type}')

    def f(x):
        """The function for root finding."""
        n, k = x
        t_calc, r_calc = calc_T_and_R(n, k, n_sub, k_sub, wl)
        return t_calc - t_meas, r_calc - r_meas

    # Polish roots found with the graphical method with NR
    for root_index in range(len(roots)):
        res = root(f, np.asarray(roots[root_index]))
        if res.success:
            all_roots.append([wl, res.x[0], res.x[1]])
            print(res.x)
        else:
            pass  # most probably just false intersection (could be checked with finer grid)
            # raise Exception(f'Could not converge (index = {index}).')

# Save all the roots to the text file.
all_roots = np.array(all_roots)
np.savetxt('roots.txt', all_roots)

# Create figure.
fig, ax_n = plt.subplots(1, 1, constrained_layout=True)
ax_k = ax_n.twinx()
fig.canvas.manager.set_window_title('figure')

# Configure axes.
ax_n.set_xlabel(r'Wavelength (nm)')
ax_n.set_ylabel(r'n')
ax_k.set_ylabel(r'k')

# Plot n and k.
l_n, = ax_n.plot(all_roots[:, 0], all_roots[:, 1], '.', ms=7, mec='none', c=COLORS[0], alpha=0.9, label='n')
l_k, = ax_k.plot(all_roots[:, 0], all_roots[:, 2], '.', ms=7, mec='none', c=COLORS[1], alpha=0.9, label='k')

# Create legend.
ax_n.legend(handles=[l_n, l_k], loc='center right')

# Show the title.
ax_n.set_title('GST')

# Show the window.
plt.show()
