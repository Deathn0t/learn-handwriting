from multicouche_v3 import *
from neurone_v3 import *
from math import sin, pi
import pylab as py
import numpy as np

C1 = Couche("C1", 1, Sigmoid, [Neurone(str(i), 1, Sigmoid) for i in range(20)])
C2 = Couche("C2", 20, Sigmoid, [Neurone(str(i), 20, Sigmoid) for i in range(10)])
C3 = Couche("C3", 10, Sigmoid, [Neurone(str(i), 10, Sigmoid) for i in range(20)])
C4 = Couche("C2", 20, Sigmoid, [Neurone(str(i), 20, Sigmoid) for i in range(1)])
Br = Cerveau("Br", Sigmoid, [C1, C2, C3, C4])

#f = lambda x: (x*((1+sin(x*6*pi)))/2.)
#f = lambda x: ((1-sin(x*4)*sin(x*4)))
g = lambda x: 1/float(1.+(25*x*x))
f = lambda x: g(x-1./2.)
h = lambda x: h(float(x)/2.)
lx = [i/300. for i in range(300)]
ly = [f(i) for i in lx]
modele = [(np.array([x]), np.array([y])) for x,y in zip(lx,ly)]


# update a distribution based on new data.
import numpy as np
from matplotlib.animation import FuncAnimation


class UpdateF(object):
    def __init__(self, ax, cerveau, modele):
        self.iter = 0
        self.success = 0
        self.cerveau = cerveau
        self.modele = modele
        self.serie = [i for i in range(len(self.modele))]
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 300)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True)

    def init(self):
        self.success = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        self.iter += 1
        print(self.iter)
        if i == 0:
            return self.init()
        pas = 1.
        if self.iter%1000 == 0:
            pas = pas*0.1
        gradLearn(self.cerveau, self.modele, self.serie, pas)
        y = [self.cerveau.influx_cerveau(np.array([r])) for r in self.x]
        self.line.set_data(self.x, y)
        return self.line,

fig, ax = py.subplots()
ud = UpdateF(ax, Br, modele)

anim = FuncAnimation(fig, ud, frames=np.arange(10000), init_func=ud.init,
                     interval=10, blit=True)
py.plot(lx,ly)
py.show()