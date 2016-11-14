import os
import numpy
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def parse(line):
    size, threads, time = line.split()
    return int(size), int(threads), float(time)


values = []
for name in os.listdir('results'):
    with open(os.path.join('results', name)) as res_file:
        values += [parse(line) for line in res_file.readlines()]
values.sort()
values = numpy.asarray(values).reshape((-1, 8, 3))
size, threads, time = [values[:, :, i] for i in range(3)]

figure= pyplot.figure(figsize=(10, 6))
axes = figure.gca(projection='3d')
axes.set_xlabel('Threads')
axes.set_ylabel('Matrix size')
axes.set_zlabel('Time (s)')
surface = axes.plot_surface(threads, size, time, rstride=1, cstride=1, 
                            cmap=cm.rainbow, linewidth=0)
figure.colorbar(surface, shrink=0.5)
pyplot.savefig('performance.png', format='png', dpi=300)
