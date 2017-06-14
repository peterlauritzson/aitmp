
import matplotlib.pyplot as plt
import autograd.numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

import PSO
from FitnessFunctions import beale_function as f


MINIMUM = [3, 0.5] #None
minima = np.array(MINIMUM).reshape(-1, 1)


class TrajectoryAnimation3D(animation.FuncAnimation):
    def __init__(self, *paths, zpaths, labels=[], fig=None, ax=None, frames=None,
                 interval=100, repeat_delay=5, blit=True, repeat=False, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax

        self.paths = paths
        self.zpaths = zpaths

        if frames is None:
            frames = max(paths[0].shape)

        self.lines = [ax.plot([], [], 'b*')[0]
                      for _, label in zip_longest(paths, labels)]

        super(TrajectoryAnimation3D, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                    frames=frames, interval=interval, blit=blit,
                                                    repeat_delay=repeat_delay, repeat=repeat, **kwargs)

    def init_anim(self):
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return self.lines

    def animate(self, i):
        for line, path, zpath in zip(self.lines, self.paths, self.zpaths):
            line.set_data(*path[i])
            line.set_3d_properties(zpath[i])
        return self.lines


xmin, xmax, xstep = -10, 10, .2
ymin, ymax, ystep = -10, 10, .2

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f([x, y])

fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

paths = PSO.run_algorithm(save_particles=True)
zpaths = list()
for path in paths:
    zhist = list()
    for position in path:
        zhist.append(f(position))
    zpaths.append(zhist)

paths = np.array(paths)
zpaths = np.array(zpaths)

if MINIMUM is not None:
    ax.plot(*minima, f(minima), 'r*', markersize=10)

anim = TrajectoryAnimation3D(*paths, zpaths=zpaths, ax=ax)

plt.show()


