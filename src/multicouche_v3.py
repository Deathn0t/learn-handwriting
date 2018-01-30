import numpy as np
import pylab as py
from neurone_v3 import *


def apprentissage_wh(cerveau, modele, epsilon):
    etr, sTh = modele[0], modele[1]
    mat = cerveau.influx_app(etr)
    err = [np.array([0. for elt in lg]) for lg in mat]
    for i in range(len(mat)-1, -1, -1):
        for j in range(len(mat[i])-1, -1, -1):
            if (i == len(mat)-1):
                err[i][j] = mat[i][j] * (1-mat[i][j]) * (sTh[j] - mat[i][j])
            else:
                som = 0
                lc = cerveau.reseau[i].couche
                for k in range(len(lc)):
                    som = lc[k].vect_w[j] * err[i+1][k]
                err[i][j] = mat[i][j] * (1 - mat[i][j]) * som
    for i in range(len(mat)-2, -1, -1):
        for j in range(len(mat[i])-1, -1, -1):
            lc = cerveau.reseau[i].couche
            for k in range(len(lc)):
                lc[k].vect_w[j] = lc[k].vect_w[j] + 0.5* err[i+1][k]*mat[i][j]

def apprentissage_gradient(cerveau, m, epsilon) :
    """Algorithme d'apprentissage d'un reseau de neurone
    dont la fonction, d'activation est x -> 1/{1+e^{-x}}."""
    lg_cerveau = len(cerveau.reseau)
    delta = np.array([np.array([0. for neurone in cerveau.reseau[i].couche]) for i in range(lg_cerveau)])
    (x, mx) = m
    matrice = cerveau.influx_app(x)
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

def gradLearn(cerveau, modele, serie, epsilon):
    for i in serie:
        apprentissage_gradient(cerveau, modele[i], epsilon)

def egaliteV(v1,v2):
    if (np.shape(v1) == np.shape(v2)):
        for i in range(len(v1)):
            if v1[i] != v2[i]:
                return False
        return True
    else:
        return False

def arrondiS(l, s =0.5):
    "applique un effet : arrondi les valeurs a 1. si >= s et a 0. sinon"
    for i in range(len(l)):
            l[i] = 1. if (l[i] >= s) else 0.

def arrondiM(l):
    pos, m = maxL(l)
    for i in range(len(l)):
        l[i] = 1. if (i == pos) else 0.

def maxL(l):
    m = l[0]
    pos = 0
    for i,elt in enumerate(l):
        if elt > m:
            pos, m = i, elt
    return (pos, m)
    
def minL(l):
    m = l[0]
    pos = 0
    for i,elt in enumerate(l):
        if elt > m:
            pos, m = i, elt
    return (pos, m)

def bonnes_reponses(cerveau, modele, arrondi = arrondiM):
    b_rep = 0
    for entree,sTh in modele:
        sT = cerveau.influx_cerveau(entree)
        arrondi(sT)
        if (egaliteV(sTh, sT)):
            b_rep += 1
    return float(b_rep)/float(len(modele))*float(100)

def reponses(cerveau, modele, arrondi =arrondiM, bl = False):
    "retourne la liste des indices mals appris"
    s1 = []
    for i,couple in enumerate(modele):
        entree,sTh = couple
        sT = cerveau.influx_cerveau(entree)
        arrondi(sT)
        if not( not(egaliteV(sTh, sT)) == bl):
            s1.append(i)
    return s1

def certitude(cerveau, elt):
    entree, sTh = elt
    sortie = cerveau.influx_cerveau(entree)
    s = 0
    for e in sortie:
        s = s + e
    return s

def certitude_serie(cerveau, modele, serie):
    l = []
    for elt in serie:
        l.append(certitude(cerveau, modele[elt]))
    return l
        

def cout(cerveau, elt):
    entree, sTh = elt
    sT = cerveau.influx_cerveau(entree)
    s = 0
    for i,j in zip(sTh, sT):
        tmp = i - j
        s = s + tmp*tmp
    return s/2.

def cout_totale(cerveau, modele):
    s = 0
    for elt in modele:
        s = s + cout(cerveau,elt)
    return s

def test(a=1., n=100):
    N = Neurone("N",1,Sigmoid)
    C = Couche("C",1,Sigmoid,[N])
    B = Cerveau("B",Sigmoid,[C])
    m1 = [([1.],[0.]),
      ([0.],[1.]),
      ([10.],[1.])]
    lw = np.linspace(-2, 2, n)
    lc = []
    for w in lw:
        N.vect_w[0], N.biais = w, w
        lc.append(cout_totale(B,m1))
    py.plot(lw,lc,"b")
    N.vect_w[0], N.biais = 2., 2.
    lx = [N.vect_w[0]]
    ly = [cout_totale(B,m1)]
    for i in range(100):
        apprentissage_gradient(B,m1[0],a)
        lx.append(N.vect_w[0])
        ly.append(cout_totale(B,m1))
    py.plot(lx,ly,"r*")
    py.xlim(-2,2)
    py.ylim(0,1)
    py.show()

def tt(w,b):
    a = 1.
    N = Neurone("N",1,Sigmoid)
    C = Couche("C",1,Sigmoid,[N])
    B = Cerveau("B",Sigmoid,[C])
    N.vect_w[0], N.biais = w, b
    m1 = [([1.],[0.]),
          ([0.],[1.]),
          ([10.],[1.])]
    return cout_totale(B,m1)

"""
ttv = np.vectorize(tt)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(-2*np.pi, 2*np.pi, 100)
v = np.linspace(-2*np.pi, 2*np.pi, 100)


x = 10*np.outer(np.cos(u), np.sin(v))
y = 10*np.outer(np.sin(v), np.cos(u))

x = np.linspace(-5,5,200)
y = np.linspace(-5,5,200)
X,Y = np.meshgrid(x,y)
Z = ttv(X,Y)


ax.plot_surface(X, Y, Z,color='b')

m1 = [([1.],[0.]),
      ([0.],[1.]),
      ([10.],[1.])]
N = Neurone("N",1,Sigmoid)
C = Couche("C",1,Sigmoid,[N])
B = Cerveau("B",Sigmoid,[C])
N.vect_w[0] = 2.
N.biais = 2.
lx,ly,lz = [N.vect_w[0]], [N.biais], [cout_totale(B,m1)]
for i in range(100):
    apprentissage_gradient(B,m1[0],1.)
    lx.append(N.vect_w[0])
    ly.append(N.biais)
    lz.append(cout_totale(B,m1))

ax.plot(lx,ly,lz,"r*")

plt.show()"""
