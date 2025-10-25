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
DATA_DIR = 'C:\\Users\\juriy\\Documents\\MEGA\\Projects\\GST spectroscopy\\data\\2025-07'
R_FILENAME = 'R_Glass_clean(1737f).csv'
T_FILENAME = 'T_Glass_clean(1737f).csv'
rt = RTPair.from_ftir_files(os.path.join(DATA_DIR, R_FILENAME), os.path.join(DATA_DIR, T_FILENAME))
rt.strip(0, 4000)

# Print a bit of an overview.
print(f'R spans from {rt.R.min():.4f} to {rt.R.max():.4f}')
print(f'T spans from {rt.T.min():.4f} to {rt.T.max():.4f}')
print(f'Maximum R + T is {(rt.R + rt.T).max():.4f}')
rt.plot()
