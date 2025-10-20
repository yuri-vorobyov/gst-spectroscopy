"""
This script plots raw R and T spectra together with their sum.

Intended for first checking of experimental data.
"""
from RTPair import RTPair


# Load data.
rt = RTPair.from_ftir_files(r'C:\Users\juriy\Documents\MEGA\Projects\GST spectroscopy\data\2025-07\R_Glass_clean(1737f).csv',\
                            r'C:\Users\juriy\Documents\MEGA\Projects\GST spectroscopy\data\2025-07\T_Glass_clean(1737f).csv')
rt.strip(0, 8000)

# Print a bit of an overview.
print(f'R spans from {rt.R.min():.4f} to {rt.R.max():.4f}')
print(f'T spans from {rt.T.min():.4f} to {rt.T.max():.4f}')
print(f'Maximum R + T is {(rt.R + rt.T).max():.4f}')
rt.plot()
