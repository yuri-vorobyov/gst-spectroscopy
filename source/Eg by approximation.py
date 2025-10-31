"""
Original name: "Eg by approximation.py"

Calculate Eg from the spectrum of optical constants using Tauc plot.

Usage:
1. Make sure `ocs` points to the right file.
2. Run the script.
3. Once the plot window appears --- use mouse to select the linear range. Approximation
   results will be displayed.
4. Repeat if necessary.
"""

import os
from pathlib import Path
import numpy as np
from OpticalConstantsSpectrum import OpticalConstantsSpectrum as OCS
from pathlib import Path
from matplotlib.widgets import SpanSelector

os.chdir(Path(__file__).parent.resolve())

# Load data.
ocs = OCS.from_wnk_file(f"physical roots.txt")

# Make the Tauc plot.
ax = ocs.plot(scale="Tauc")

# A hack below. Matplotlib keeps redrawing plot extending the viewport to the x = 0
# after approximation line is added. Two lines below prevent such behavior.
x_min, _ = ax.get_xlim()
ax.set_xlim(x_min, None)


def onselect(xmin, xmax):
    this = onselect
    # Define state variables as function attributes.
    if not hasattr(this, "line"):
        this.line = None
    if not hasattr(this, "vspan"):
        this.vspan = None
    if not hasattr(this, "text"):
        this.text = None
    # Extract points selected to participate in the approximation.
    ii = (ocs.e > xmin) * (ocs.e < xmax)
    selected_x, selected_y = ocs.e[ii], (ocs.alpha[ii] * ocs.e[ii]) ** 0.5
    if len(selected_x) == 0:
        return
    # Highlight approximation region.
    if this.vspan:
        this.vspan.remove()
    this.vspan = ax.axvspan(
        xmin, xmax, facecolor="green", alpha=0.2, linewidth=1.0, edgecolor="darkgreen"
    )
    # Compute approximating line and draw it.
    res = np.polyfit(selected_x, selected_y, deg=1)
    if this.line:
        this.line.remove()
    (this.line,) = ax.plot(ocs.e, res[1] + res[0] * ocs.e, "-k", lw=1.5)
    # Display Eg value.
    if this.text:
        this.text.remove()
    this.text = ax.text(
        0.05,
        0.85,
        f"$ E_g \\, = \\, {-res[1] / res[0]:.3f} \\; \\mathrm{{eV}} $",
        transform=ax.transAxes,
    )


span = SpanSelector(ax, onselect, "horizontal", props=dict(alpha=0.6, facecolor="cyan"))

ocs.show()
