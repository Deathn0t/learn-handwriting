from random import *
import matplotlib.pyplot as py

###Cinema###
# 0 : sante bonne
# 0 : meteo bonne
# 0 : voiture disponible
# 0 : bon film

conditions = ["sante bonne", "meteo bonne",
              "voiture disponible", "bon film"]
modele = [([0,0,0,0], 0),
          ([1,0,0,0], 0),
          ([0,1,0,0], 0),
          ([0,0,1,0], 0),
          ([0,0,0,1], 0),
          ([1,1,0,0], 0),
          ([1,0,1,0], 0),
          ([1,0,0,1], 1),
          ([0,1,1,0], 0),
          ([0,1,0,1], 1),
          ([0,0,1,1], 0),
          ([1,1,1,0], 1),
          ([1,1,0,1], 1),
          ([1,0,1,1], 1),
          ([0,1,1,1], 1),
          ([1,1,1,1], 1)]
    
def prod_scal(x,y):
    n = len(x)
    s = 0
    for i in range(n):
        s = s + x[i]*y[i]
    return s

def seuil(x):
    return 0 if (x < 0) else 1

def aleatoire(modele, taille):
    alea = [i for i in range(len(modele))]
    l = []
    for i in range(taille):
        x = randrange(0, 16)
        while not(x in alea):
            x = randrange(0, 16)
        alea.remove(x)
        l.append(modele[x])
    return l

def apprentissage(modele,neurone, S, alpha =1.):
    #d, f = 0,16
    condition = 0
    while condition != S:
        for entree,sortie_th in modele:
            #entree, sortie_th = modele[d:f][randrange(0,f-d)]
            sortie = neurone.influx(entree)
            if (sortie != sortie_th):
                delta = alpha*(sortie_th-sortie)
                neurone.biais = neurone.biais + delta#############################
                for i in range(neurone.ordre):
                    neurone.vect_w[i] = neurone.vect_w[i] + (delta * entree[i])
        condition += 1

#class Neurone(object):
#    
#    def __init__(self, nb_x):
#        self.nb_x = nb_x
#        self.vect_w = [randrange(-9,10) for i in range(nb_x)]
#        self.biais = randrange(-9,10)
#
#    def influx(self, vect_x):
#        return seuil( prod_scal(vect_x, self.vect_w) + self.biais)


def bonnes_reponses(modele, neurone):
    b_rep = 0
    for entree,sortie in modele:
        b_rep = b_rep + 1 if (neurone.influx(entree) == sortie) else b_rep
    return float(b_rep)/float(len(modele))*float(100)        

NUM = 1

def courbes(n,iteration,mod):
    lx = [i for i in range(iteration)]
    #conditions_finales = [[i] for i in range(n)]
    #alpha_liste = [0.01+ (0.5*i) for i in range(5)]
    for j in range(n):
        N1 = Neurone(4)
        #N1.vect_w = [4, 2, -3, 1]
        #N1.biais = 6
        #conditions_finales[j].append(N1.vect_w[:])
        #conditions_finales[j].append(N1.biais)
        ly = []
        #mod = aleatoire(modele, 5)
        #print(mod)
        for i in lx:
            apprentissage(mod,N1,0.5)
            ly.append(bonnes_reponses(modele,N1))
        py.plot(lx,ly)
        #conditions_finales[j].append(N1.vect_w[:])
        #conditions_finales[j].append(N1.biais)
    #return conditions_finales
    #return alpha_liste

###Programme###

cf = courbes(100,100)
#py.legend(cf, loc=4, borderaxespad=0.)
py.show()

