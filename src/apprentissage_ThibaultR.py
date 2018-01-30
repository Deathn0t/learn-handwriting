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
# Test 1 : cerveau fixe, nb d'iterations fixe, taille modele variable
#==============================================================================

def f(x) :
    return 1/(1+exp(-x))

g = Genome("g", f)    

# Cerveau enregistre dans experience1 et experience2

K1 = Couche("K1", 1, g, [Neurone("",1,g) for i in range(100)])
K2 = Couche("K2", 100, g, [Neurone("",100,g) for i in range(1)])

S = Cerveau("S", g, [K1,K2])

def experience1(z) :
    lx = [x/200 for x in range(201)]
    ly = [z(x) for x in lx]
    py.plot(lx, ly, label = "Courbe reelle")
    for k in range(1,11) :
        R = S
        modele = [(np.array([x/float((5* k))]),np.array([z(x/float((5* k)))])) for x in range(5*k + 1)]
        for i in range(1000) :
            if i% 100 == 0 :
                print(k,i)
            apprentissage_gradient(R, modele, 1.)
        ly = [R.influx_cerveau(np.array([x])) for x in lx]
        s = str(k)
        py.plot(lx,ly, label = s)
    py.legend()
    py.show()
    
def experience1_carre() :
    def z(x) :
        return x**2
    experience1(z)

def experience1_sinus() :
    def z(x) :
        return 1 - (sin(2*x))**2
    experience1(z)
    
def experience1_exp() :
    def z(x) :
        return (exp(x) - 1)/(exp(1.) - 1)
    experience1(z)

experience1_sinus()

#==============================================================================
# Test 2 : cerveau fixe, nb d'iterations variable, taille modele fixe
#==============================================================================

def experience2(z) :
    lx = [x/200 for x in range(201)]
    ly = [z(x) for x in lx]
    py.plot(lx, ly, label = "Courbe reelle")
    for k in range(1,11) :
        R = Cerveau()
        R.generer("cerveaux/experience1", g)
        modele = [(np.array([x/50]),np.array([z(x/50)])) for x in range(51)]
        for i in range(100* k) :
            if i% 100 == 0 :
                print(k,i)
            apprentissage_gradient(R, modele, 0.5)
        ly = [R.influx_cerveau(np.array([x])) for x in lx]
        s = str(k)
        py.plot(lx,ly, label = s)
    py.legend()
    py.show()
    
def experience2_carre() :
    def z(x) :
        return x**2
    experience2(z)

def experience2_sinus() :
    def z(x) :
        return 1 - (sin(2*x))**2
    experience2(z)
    
def experience2_exp() :
    def z(x) :
        return (exp(x) - 1)/(exp(1.) - 1)
    experience2(z)