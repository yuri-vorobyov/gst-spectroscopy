"""
Original name: "film on substrate inversion.py"

Calculate n and k spectra of a thin film from R&T spectra provided the substrate n and k
are known.

Usage:
1. Make sure `DATA_DIR`, `R_FILENAME`, and `T_FILENAME` point to the right places.
2. Make sure `sub_nk` points to the file from `substrate inversion.py` script.
3. Set `d_film` to the known film thickness.
4. Run the script. Wait until window with roots is shown. Roots are saved to the text
   file.
5. Adjust the film thickness until the picture of roots is physically sound (has
   characteristic intersection).
"""

import os
from pathlib import Path
from RTPair import RTPair
from OpticalConstantsSpectrum import OpticalConstantsSpectrum as OCS
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import contourpy
import shapely
import itertools
from calc import calc_RT_AFSA

os.chdir(Path(__file__).parent.resolve())

plt.style.use("style.mplstyle")
plt.rcParams["savefig.directory"] = "."
COLORS = [item["color"] for item in plt.rcParams["axes.prop_cycle"].__dict__["_left"]]

np.set_printoptions(precision=4)

# Load thin-film sample spectrum data.
DATA_DIR = Path("../test data")
R_FILENAME = "R_GeTe_4819(130nm)_chocolate_bar.csv"
T_FILENAME = "T_GeTe_4819(130nm)_chocolate_bar.csv"
film_rt = RTPair.from_ftir_files(DATA_DIR / R_FILENAME, DATA_DIR / T_FILENAME)
film_rt.strip(550, 2250)
film_rt.resample()

# Load optical constants spectrum of substrate material.
sub_nk = OCS.from_wnk_file("substrate (w,n,k).txt")

# Check if resampling is needed.
if len(sub_nk.w) == len(film_rt.w) and np.allclose(sub_nk.w, film_rt.w, rtol=1e-6):
    print(
        "Spectra of substrate and thin film have the same scale --- "
        "no resampling needed."
    )
else:
    print(
        "Spectra of substrate and thin film have different scales. "
        "Resample for overlapped region."
    )
    w_min, w_max = max(sub_nk.w[0], film_rt.w[0]), min(sub_nk.w[-1], film_rt.w[-1])
    step = 0.5  # Step is fixed to 0.5 nm for the sake of simplicity.
    scale = np.linspace(w_min, w_max, step)
    film_rt.resample(scale)
    sub_nk.resample(scale)

# Thicknesses are known.
d_film = 147.6  # nm
d_sub = 0.7e-3 * 1e9  # nm

# Graphical method is implemented as follows. For each trial pair of values of n and k
# within the limits both T_meas - T_calc and R_meas - R_calc. Therefore, here we need
# the scale of possible n and k values.
lim_n, lim_k = [-0.2, 6.0], [-0.2, 3.0]
N_n, N_k = 30, 15
n_trial, k_trial = np.meshgrid(
    np.linspace(lim_n[0], lim_n[1], N_n),
    np.linspace(lim_k[0], lim_k[1], N_k),
    indexing="ij",
)

# Prepare containers for calculated values.
T_trial = np.empty((N_n, N_k))
R_trial = np.empty((N_n, N_k))


def calc_T_and_R(film_n, film_k, substrate_n, substrate_k, wavelength_nm):
    """Calculate T and R (as a tuple) of a film on a substrate."""
    r_calc, t_calc = calc_RT_AFSA(
        wavelength_nm, film_n, film_k, d_film, substrate_n, substrate_k, d_sub
    )
    return t_calc, r_calc


def update_trial_matrix(wavelength, sub_n, sub_k, measured_R, measured_T):
    T_arr, R_arr = calc_T_and_R(n_trial, k_trial, sub_n, sub_k, wavelength)
    for i in range(N_n):
        for j in range(N_k):
            T_trial[i, j], R_trial[i, j] = (
                T_arr[i, j] - measured_T,
                R_arr[i, j] - measured_R,
            )


# Calculate the solution contours. Effectively, `z` is passed by reference so change to
# corresponding array will propagate into the contour generator object automatically.
cg_R = contourpy.contour_generator(
    n_trial, k_trial, R_trial, name="serial", line_type="Separate"
)
cg_T = contourpy.contour_generator(
    n_trial, k_trial, T_trial, name="serial", line_type="Separate"
)

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
for index, wl in enumerate(film_rt.w):
    n_sub, k_sub = sub_nk.n[index], sub_nk.k[index]
    t_meas, r_meas = film_rt.T[index], film_rt.R[index]
    print(
        f"{index:>4} solving for {wl:.1f} nm, n_sub = {n_sub:.3f}, k_sub = {k_sub:.3g}"
    )

    update_trial_matrix(wl, n_sub, k_sub, r_meas, t_meas)

    roots_R = cg_R.lines(0.0)  # list of contour parts
    roots_T = cg_T.lines(0.0)  # list of contour parts

    # Find all intersection points, which are estimates of the solutions.
    roots = []
    for lsR, lsT in map(
        lambda x: map(shapely.LineString, x), itertools.product(roots_R, roots_T)
    ):
        if lsR.intersects(lsT):
            intersection = shapely.intersection(lsR, lsT)
            if intersection.geom_type == "MultiPoint":
                roots.extend(map(lambda x: (x.x, x.y), intersection.geoms))
            elif intersection.geom_type == "Point":
                roots.append((intersection.x, intersection.y))
            else:
                raise Exception(f"Dont know what to do with {intersection.geom_type}")

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
            pass  # most probably just false intersection (check with finer grid)
            # raise Exception(f'Could not converge (index = {index}).')

# Save all the roots to the text file.
all_roots = np.array(all_roots)
np.savetxt(f"all roots.txt", all_roots)
print("saved")

nk = OCS(all_roots[:, 0], all_roots[:, 1], all_roots[:, 2])
nk.plot(scale="wavelength")
nk.show()
