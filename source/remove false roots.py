"""
remove points.py

This script allows one to manually select points on a plot and remove them.
"""


import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, axes, collection, alpha_other=0.3):
        self.axes = axes
        self.canvas = self.axes.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor property.')
        # Set default color
        self.default_color = [0.12156863, 0.46666667, 0.70588235, 0.6]
        # And "selected" color
        self.selected_color = [0.7, 0.84, 0.64, 0.3]
        self.fc = np.tile(self.default_color, (self.Npts, 1))
        self.collection.set_facecolor(self.fc)
        self.collection.set_edgecolor('none')

        self.lasso = LassoSelector(ax, onselect=self.on_select)

        # Indexes of selected points.
        self.ind = []

        # state of specific keys
        self.key_control = False
        self.key_shift = False

        self.keyPress = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.keyRelease = self.canvas.mpl_connect('key_release_event', self.on_key_release)

    def delete_selected(self):
        """Remove selected points, if any."""
        ii = self.collection.get_facecolor()[:, -1] > 0.3  # locate points to be removed
        self.xys = self.collection.get_offsets()[ii]
        self.collection.set_offsets(self.xys)
        self.Npts = len(self.xys)
        self.fc = np.tile(self.default_color, (self.Npts, 1))
        self.collection.set_facecolor(self.fc)
        selector.canvas.draw_idle()

    def selected(self):
        """Return indexes of selected points."""
        pass

    def unselected(self):
        """Return mask array of unselected points."""
        # Read color for each point (including alpha value).
        colors = self.collection.get_facecolors()
        if len(colors) == 1:  # in case all the points have the same color
            colors = np.tile(colors, (self.Npts, 1))
        ii = self.collection.get_facecolor()[:, -1] > 0.3  # unselected points
        return ii

    def on_select(self, vertices):
        # Save color for each point (including alpha value).
        colors = self.collection.get_facecolors()
        if len(colors) == 1:  # in case all the points have the same color the single color is returned
            colors = np.tile(colors, (self.Npts, 1))
        # Assemble lasso path.
        path = Path(vertices)
        # Get indexes of selected points.
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        # Normal operation (control and shift keys are released) --- set as selected.
        if (not self.key_control) and (not self.key_shift):
            colors[self.ind] = self.selected_color
        # Update
        self.collection.set_facecolors(colors)
        self.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 'control':
            self.key_control = True
        if event.key == 'shift':
            self.key_shift = True

    def on_key_release(self, event):
        if event.key == 'control':
            self.key_control = False
        if event.key == 'shift':
            self.key_shift = False

    def disconnect(self):
        """ Disable selecting capability. """
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('style.mplstyle')

    data = np.loadtxt('roots.txt')  # roots are (w, n, k) tuples

    fig, ax = plt.subplots()
    pts = ax.scatter(data[:, 1], data[:, 2], s=20)
    ax.set_xlabel('n')
    ax.set_ylabel('k')

    selector = SelectFromCollection(ax, pts)

    print('Draw a line around the points you want to remove.\nThey will be marked.\nRepeat if necessary.\nUse zoom '
          'and pan tools for precision, but don\'t forget to disable them afterwards.\n"Enter" key saves points which '
          'were not selected to "roots.txt".')

    def accept(event):
        if event.key == "enter":
            np.savetxt('roots.txt', data[selector.unselected()])
            print('saved!')

    fig.canvas.mpl_connect("key_press_event", accept)

    plt.show()
