"""
Original name: "plot R+T.py"

This script plots raw R and T spectra.

Intended for first checking of experimental data.

Usage:
1. Change `DATA_DIR` variable in accordance with the location of data files.
2. Change `R_FILENAME` and `T_FILENAME` as well.
3. Run script.
4. Make sure that R+T does not exceed unity. Otherwise, spectrum data must be corrected somehow.
"""

from RTPair import RTPair
import os.path


# Load data.
DATA_DIR = 'C:/Users/juriy/Documents/MEGA/Projects/GST spectroscopy/data/2025-07'
R_FILENAME = 'R_GeTe_4819(130nm)_chocolate_bar.csv'
T_FILENAME = 'T_GeTe_4819(130nm)_chocolate_bar.csv'
rt = RTPair.from_ftir_files(os.path.join(DATA_DIR, R_FILENAME), os.path.join(DATA_DIR, T_FILENAME))
rt.strip(550, 2250)
rt.resample()

# Print a bit of an overview.
print(f'R spans from {rt.R.min():.4f} to {rt.R.max():.4f}')
print(f'T spans from {rt.T.min():.4f} to {rt.T.max():.4f}')
RpT = rt.R + rt.T
i_max = RpT.argmax()
print(f'Maximum R + T is {RpT[i_max]:.4f}')
print(f'              at {rt.w[i_max]:.1f} nm')
print(f'{len(rt.w)} points')
print(f'Step is from {rt.w[1] - rt.w[0]:.4f} nm to {rt.w[-1] - rt.w[-2]:.4f} nm')


rt.plot()
