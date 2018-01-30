from time import *
import numpy as np
from multicouche_v3 import *
from neurone_v3 import *
from math import sin, pi
import pylab as py
r = 100
C1 = Couche("C1", 1, Sigmoid, [Neurone(str(i), 1, Sigmoid) for i in range(r)])
C2 = Couche("C2", r, Sigmoid, [Neurone(str(i), r, Sigmoid) for i in range(r)])
C3 = Couche("C3", r, Sigmoid, [Neurone(str(i), r, Sigmoid) for i in range(1)])
C4 = Couche("C2", 20, Sigmoid, [Neurone(str(i), 20, Sigmoid) for i in range(1)])
Br = Cerveau("Br", Sigmoid, [C1, C2, C3])

f = lambda x: (x*((1+sin(x*6*pi)))/2.)
lx = [i/100. for i in range(100)]
ly = [f(i) for i in lx]
modele = [(np.array([x]), np.array([y])) for x,y in zip(lx,ly)]
mr =  [(np.array([1.]), np.array([1.])) for i in range(1)]
l = lambda x: apprentissage_gradient(Br, mr, x)

####Calcul####
def ct(f, a,  n):
    t1 = time()
    for i in range(n):
        f(a)
    t2 = time()
    return t2-t1

print(ct(l, 1., 10000))
