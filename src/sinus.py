from random import *
from math import *
from neurone_v3 import *
import pylab as py
import numpy as np

def apprentissage_gradient(cerveau, modele, epsilon) :
    """Algorithme d'apprentissage d'un reseau de neurone
    dont la fonction, d'activation est x -> 1/{1+e^{-x}}."""
    lg_cerveau = len(cerveau.reseau)
    delta = np.array( [ np.array([0. for neurone in cerveau.reseau[i].couche]) for i in range(lg_cerveau)])
    for m in modele :
        (x, mx) = m
        matrice = cerveau.influx_app( x)
        i = 0
        for neurone in cerveau.reseau[lg_cerveau - 1].couche :
            delta[lg_cerveau-1][i] =   matrice[lg_cerveau-1][i] * (1 - matrice[lg_cerveau-1][i]) * (mx[i] - matrice[lg_cerveau-1][i])
            i = i + 1
        for k in range(lg_cerveau-1) :
            i = 0
            for neurone in cerveau.reseau[lg_cerveau-2-k].couche :
                s = 0.
                j = 0
                for cellule in cerveau.reseau[lg_cerveau-1-k].couche :
                    s = s +  delta[lg_cerveau-1-k][j] * cellule.vect_w[i]
                    j = j + 1
 
                delta[lg_cerveau-2-k][i] =  matrice[lg_cerveau-2-k][i] * (1- matrice[lg_cerveau-2-k][i]) * s
                i = i + 1
        for i in range(len(cerveau.reseau)) :
            for j in range(len(cerveau.reseau[i].couche)) :
                cerveau.reseau[i].couche[j].biais = cerveau.reseau[i].couche[j].biais + epsilon * delta[i][j]
                for k in range(len(cerveau.reseau[i].couche[j].vect_w)) :

                    if i == 0 :
                        e = x[k]
                    else :
                        e = matrice[i-1][k]
                    cerveau.reseau[i].couche[j].vect_w[k] = cerveau.reseau[i].couche[j].vect_w[k] + epsilon * delta [i][j] * e
                    
#==============================================================================
# Test
#==============================================================================

def f(x) :
    return 1/(1+exp(-x))

g = Genome("g", f)    

N1_1 = Neurone("N1_1", 1, g)
N1_2 = Neurone("N1_2", 1, g)
N1_3 = Neurone("N1_3", 1, g)

C1 = Couche("C1", 1, g, [N1_1, N1_2, N1_3])

N2_1 = Neurone("N2_1", 3, g)
N2_2 = Neurone("N2_2", 3, g)
N2_3 = Neurone("N2_3", 3, g)
N2_4 = Neurone("N2_4", 3, g)
N2_5 = Neurone("N2_5", 3, g)


C2 = Couche("C1", 3, g, [N2_1, N2_2, N2_3, N2_4, N2_5])

N3 = Neurone("N3", 5, g)

C3 = Couche("C3", 5, g, [N3])

R = Cerveau("R", g, [C1,C2,C3])

modele = [(np.array([x/200]),np.array([(x/200)**2])) for x in range(201)]

print(R.influx_cerveau(np.array([0.25])))

for i in range(1000) :
    apprentissage_gradient(R, modele, 0.5)

print(R.influx_cerveau(np.array([0.25])))

lx = [x/200 for x in range(201)]
ly = [R.influx_cerveau(np.array([x])) for x in lx]
lz = [x**2 for x in lx]
py.plot(lx,ly)
py.plot(lx,lz)
py.show()