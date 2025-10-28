"""
Original name: "Eg by histogram.py"

Script estimates Eg from Tauc plot using histogram of tangential lines at each point.

Usage:
1. Make sure data is loaded from the right file.
2. Run script.
3. Wait until the plot is shown. Make sure approximation line fits data correctly.
4. Take Eg value from the console.
"""

import numpy as np
from OpticalConstantsSpectrum import OpticalConstantsSpectrum as OCS
from pathlib import Path
from sg_smooth.smoothing import smSGtan_bisquare


def fd_bins(data):
    """Freedman-Diaconis bins number."""
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    h = 2 * iqr / (len(data) ** (1/3))
    return int(np.ceil((data.max() - data.min()) / h))


# Load data.
ocs = OCS.from_wnk_file(f'physical roots.txt')

# Compute tangentail lines at each point.
_, slope, inter = smSGtan_bisquare(ocs.e, (ocs.alpha * ocs.e)**0.5, 25, 3)

# Compute coarse 2D histogram of tangent line parameters.
counts, xedges, yedges = np.histogram2d(slope, inter, bins=(fd_bins(slope), fd_bins(inter)))
# Locate the maximum (the range of bin having maximal `count` of points).
xi, yi = np.unravel_index(counts.argmax(), counts.shape)
range = [(xedges[xi], xedges[xi + 1]), (yedges[yi], yedges[yi + 1])]
# Now refined histogram.
counts, xedges, yedges = np.histogram2d(slope, inter, bins=10, range=range)
# Its maximum.
xi, yi = np.unravel_index(counts.argmax(), counts.shape)
print(f'In the maximal bin of the refined histogram there are {int(counts[xi, yi])} tangents.')
slope, inter = (xedges[xi + 1] + xedges[xi]) / 2, (yedges[yi + 1] + yedges[yi]) / 2
re_x = abs((xedges[xi + 1] - xedges[xi])) / abs(slope)
re_y = abs((yedges[yi + 1] - yedges[yi])) / abs(inter)
print(f'Uncertainty of slope is {re_x * 100:.2f} %, and uncertainty of intercept is {re_y * 100:.2f} %')
print(f'Eg = {-inter / slope:.4f} eV')

# Make the Tauc plot.
ax = ocs.plot(scale='Tauc')

# Linear approximation.
ax.plot(ocs.e, inter + slope * ocs.e, '-k', lw=2.0)
ax.set_ylim(0, None)

ocs.show()
