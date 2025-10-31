"""
Original name: "substrate inversion.py"

Calculate n and k spectra of substrate sample from its corresponding R&T spectra using
root-finding method. Interference is neglected. Inversion in case of single substrate
gives only one root, so the whole process is automatic.

Usage:
1. Change `DATA_DIR`, `R_FILENAME`, and `T_FILENAME` constants in accordance with the
   location of data files.
2. Change `D_SUB` constant so that its value is the substrate thickness in nanometers.
3. Run the script and wait until the n&k plot is shown.
4. Results are saved to the text file.
"""

import os
from pathlib import Path
from RTPair import RTPair
from OpticalConstantsSpectrum import OpticalConstantsSpectrum as OCS
from calc import calc_RT_ASA
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

os.chdir(Path(__file__).parent.resolve())

# Load data.
DATA_DIR = Path("../test data")
R_FILENAME = "R_Glass_clean(1737f).csv"
T_FILENAME = "T_Glass_clean(1737f).csv"
rt = RTPair.from_ftir_files(DATA_DIR / R_FILENAME, DATA_DIR / T_FILENAME)
rt.strip(550, 2250)
rt.resample()

wavelengths = rt.w

# Substrate thickness is known.
D_SUB = 0.7e-3 * 1e9  # nm

# Root-finding is used to obtain n, k pairs for each wavelength.
ns, ks = [], []  # containers for n and k values
x0 = np.asarray([1.5, 0])  # initial guess [n, k]
for w, r_meas, t_meas in zip(rt.w, rt.R, rt.T):
    print(f"solve at {w:.1f} nm, R_meas = {r_meas:.3f}, T_meas = {t_meas:.3f}")

    def f(x):
        """The function for root finding."""
        n, k = x
        r_calc, t_calc = calc_RT_ASA(w, n, k, D_SUB)
        return r_calc - r_meas, t_calc - t_meas

    # Solution.
    res = root(f, x0)
    if res.success:
        print(f"    {res.x}")
        ns.append(res.x[0])
        ks.append(res.x[1])
        x0 = res.x  # modify initial guess to improve convergence speed
    else:
        raise Exception("Could not converge.")


ocs = OCS(wavelengths, ns, ks)
ocs.plot()
ocs.show()

# Finally, save the n and k for latter use.
ocs.save("substrate (w,n,k).txt")
print("saved")
