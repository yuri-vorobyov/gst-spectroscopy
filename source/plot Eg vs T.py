import numpy as np

import matplotlib.pyplot as plt

plt.style.use('style.mplstyle')
plt.rcParams['savefig.directory'] = '.'
COLORS = [item['color'] for item in plt.rcParams['axes.prop_cycle'].__dict__['_left']]

# creating figure
fig, ax = plt.subplots(1, 1)
fig.canvas.manager.set_window_title('figure')
ax.format_coord = lambda x, y: '{:.1f} ; {:.2g}'.format(x, y)

# configuring axes
ax.set_xlabel(r'Temperature (K)')
ax.set_ylabel(r'Band gap (eV)')

Our2024 = np.array([(273.15 - 140, 0.854),
                    (273.15 - 120, 0.842),
                    (273.15 - 100, 0.851),
                    (273.15 - 80, 0.839),
                    (273.15 - 60, 0.830),
                    (273.15 - 40, 0.823),
                    (273.15 - 20, 0.814),
                    (273.15 + 0, 0.804),
                    (273.15 + 20, 0.795),
                    (273.15 + 40, 0.780),
                    (273.15 + 60, 0.771),
                    (273.15 + 80, 0.760)])
Shportko2019 = np.array([(5, 0.95),
                         (50, 0.95),
                         (100, 0.92),
                         (150, 0.90),
                         (200, 0.87),
                         (250, 0.84),
                         (300, 0.81),
                         (350, 0.78),
                         (400, 0.75)])
Rutten2015 = np.array([(10, 0.953),
                       (50, 0.938),
                       (100, 0.920),
                       (150, 0.893),
                       (200, 0.871),
                       (250, 0.842),
                       (300, 0.815),
                       (350, 0.787)])
Luckas2011 = np.array([(5, 0.92),
                       (50, 0.93),
                       (100, 0.92),
                       (150, 0.90),
                       (200, 0.87),
                       (250, 0.84),
                       (300, 0.79),
                       (350, 0.76)])

ax.plot(Our2024[:, 0], Our2024[:, 1], ls='none', marker='*', mfc='k', mec='k', ms=7, label='Our (2024)', zorder=100)
ax.plot(Shportko2019[:, 0], Shportko2019[:, 1], ls='none', marker='s', label='Shportko (2019)')
ax.plot(Rutten2015[:, 0], Rutten2015[:, 1], ls='none', marker='o', label='Rutten (2015)')
ax.plot(Luckas2011[:, 0], Luckas2011[:, 1], ls='none', marker='^', label='Luckas (2011)')

ax.legend(loc='best')

plt.show()
