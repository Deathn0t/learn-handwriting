from random import *
from math import *
from neurone_v3 import *
import pylab as py
import numpy as np

def apprentissage_gradient(cerveau, modele, epsilon) :
    """ Algorithme d'apprentissage d'un reseau de neurone
    dont la fonction, d'activation est x -> 1/{1+e^{-x}}."""
    lg_cerveau = len(cerveau.reseau)
    delta = np.array( [ np.array([0. for neurone in cerveau.reseau[i].couche]) for i in range(lg_cerveau)])
    for m in modele :
        (x, mx) = m
        matrice = cerveau.influx_app( x)
        i = 0
        for neurone in cerveau.reseau[lg_cerveau - 1].couche :
            delta[lg_cerveau-1][i] =  matrice[lg_cerveau-1][i] * (1 - matrice[lg_cerveau-1][i]) * (mx[i] - matrice[lg_cerveau-1][i])
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
                    
def apprentissage_gradient_insistance(cerveau, modele, epsilon, insistance) :
    lg_cerveau = len(cerveau.reseau)
    delta = np.array( [ np.array([0. for neurone in cerveau.reseau[i].couche]) for i in range(lg_cerveau)])
    for m in modele :
        (x, mx) = m
        matrice = cerveau.influx_app( x)
        for a in range(insistance) :
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
# Test 1 : cerveau fixe, nb d'itérations fixe, taille modèle variable
#==============================================================================

def f(x) :
    return 1/(1+exp(-x))

g = Genome("g", f)    

# Cerveau enregistré dans experience1 et experience2

K1 = Couche("K1", 1, g, [Neurone("",1,g) for i in range(100)])
K2 = Couche("K2", 100, g, [Neurone("",100,g) for i in range(50)])
K3 = Couche("K3", 50,g, [Neurone("",50,g)])

C = Cerveau("C", g, [K1,K2,K3])


def sauvegarde_exp() :
    sv_cerveau(C, "cerveaux/experience1")

def experience1(z) :
    lx = [x/200 for x in range(201)]
    ly = [z(x) for x in lx]
    py.plot(lx, ly, label = "Courbe réelle")
    for k in range(1,11,2) :
        R = Cerveau()
        R.generer("cerveaux/experience1", g)
        modele = [(np.array([x/(5* k)]),np.array([z(x/(5* k))])) for x in range(5*k + 1)]
        for i in range(2000) :
            if i% 100 == 0 :
                print(k,i)
            apprentissage_gradient(R, modele, 0.5)
        ly = [R.influx_cerveau(np.array([x])) for x in lx]
        s = str(k)
        py.plot(lx,ly, label = s)
    py.legend(loc='best')
    py.show()
    
def experience1_carre() :
    def z(x) :
        return x**2
    experience1(z)

def experience1_sinus() :
    def z(x) :
        return 1 - (sin(4*x))**2
    experience1(z)
    
def experience1_exp() :
    def z(x) :
        return (exp(x) - 1)/(exp(1.) - 1)
    experience1(z)
    
def experience1_erreur(z) :
    lx = [x/200 for x in range(201)]
    ly = [z(x) for x in lx]
    py.plot(lx,ly)
    err = [0] * 10
    for k in range(1,11,2) :
        R = Cerveau()
        R.generer("cerveaux/experience1", g)
        modele = [(np.array([x/(5* k)]),np.array([z(x/(5* k))])) for x in range(5*k + 1)]
        for i in range(2000) :
            if i% 100 == 0 :
                print(k,i)
            apprentissage_gradient(R, modele, 0.5)
        s = str(k)
        ly = [R.influx_cerveau(np.array([x])) for x in lx]
        diff = [abs(z(float(x)) - float(R.influx_cerveau(x))) for x in lx ]
        err[k] = max(diff)
        py.plot(lx,ly, label = s)
    
    py.legend(loc='best')
    py.show()    
    print(err)

def experience1_erreur_carre() :
    def z(x) :
        return x**2
    experience1_erreur(z)
    
# [0, 0.07577643676032553, 0, 0.06592541847008315, 0, 0.059659935697076394, 0, 0.05542425280562291, 0, 0.05227295925675757] 
    
def experience1_erreur_exp() :
    def z(x) :
        return (exp(x) - 1)/(exp(1.) - 1)
    experience1_erreur(z)

# [0, 0.07250368056970757, 0, 0.06991901358086342, 0, 0.0723115209015756, 0, 0.07256446637153806, 0, 0.07184236023767333]

def experience1_erreur_sinus() :
    def z(x) :
        return 1 - (sin(4*x))**2
    experience1_erreur(z)
    
# [0, 0.41901487093523104, 0, 0.2415239530950552, 0, 0.15086151132345552, 0, 0.14284310793837396, 0, 0.12740153085350833]

#==============================================================================
# Test 2 : cerveau fixe, nb d'itérations variable, taille modèle fixe
#==============================================================================

def experience2(z) :
    lx = [x/200 for x in range(201)]
    ly = [z(x) for x in lx]
    py.plot(lx, ly, label = "Courbe réelle")
    for k in range(1,11, 2) :
        R = Cerveau()
        R.generer("cerveaux/experience1", g)
        modele = [(np.array([x/50]),np.array([z(x/50)])) for x in range(51)]
        for i in range(500* k) :
            if i% 100 == 0 :
                print(k,i)
            apprentissage_gradient(R, modele, 0.5)
        ly = [R.influx_cerveau(np.array([x])) for x in lx]
        s = str(k)
        py.plot(lx,ly, label = s)
    py.legend(loc='best')
    py.show()
    
def experience2_carre() :
    def z(x) :
        return x**2
    experience2(z)

def experience2_sinus() :
    def z(x) :
        return 1 - (sin(4*x))**2
    experience2(z)
    
def experience2_exp() :
    def z(x) :
        return (exp(x) - 1)/(exp(1.) - 1)
    experience2(z)
    
#==============================================================================
# Test 3 : cerveau variable, nb d'itérations variable, taille modèle variable
#==============================================================================

# Cerveau 1

def sauvegarde_exp3_1() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(10)])
    C2 = Couche(ordre = 10, Adn = g, couche = [Neurone(ordre = 10, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2])
    sv_cerveau(B, "cerveaux/experience3_1")

# Cerveau 2

def sauvegarde_exp3_2() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(50)])
    C2 = Couche(ordre = 50, Adn = g, couche = [Neurone(ordre = 50, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2])
    sv_cerveau(B, "cerveaux/experience3_2")

# Cerveau 3

def sauvegarde_exp3_3() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(100)])
    C2 = Couche(ordre = 100, Adn = g, couche = [Neurone(ordre = 100, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2])
    sv_cerveau(B, "cerveaux/experience3_3")

# Cerveau 4

def sauvegarde_exp3_4() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(150)])
    C2 = Couche(ordre = 150, Adn = g, couche = [Neurone(ordre = 150, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2])
    sv_cerveau(B, "cerveaux/experience3_4")

# Cerveau 5

def sauvegarde_exp3_5() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(20)])
    C2 = Couche(ordre = 20, Adn = g, couche = [Neurone(ordre = 20, Adn = g) for i in range(10)])
    C3 = Couche(ordre = 10, Adn = g, couche = [Neurone(ordre = 10, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2, C3])
    sv_cerveau(B, "cerveaux/experience3_5")

# Cerveau 6

def sauvegarde_exp3_6() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(20)])
    C2 = Couche(ordre = 20, Adn = g, couche = [Neurone(ordre = 20, Adn = g) for i in range(50)])
    C3 = Couche(ordre = 50, Adn = g, couche = [Neurone(ordre = 50, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2, C3])
    sv_cerveau(B, "cerveaux/experience3_6")

# Cerveau 7

def sauvegarde_exp3_7() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(20)])
    C2 = Couche(ordre = 20, Adn = g, couche = [Neurone(ordre = 20, Adn = g) for i in range(100)])
    C3 = Couche(ordre = 100, Adn = g, couche = [Neurone(ordre = 100, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2, C3])
    sv_cerveau(B, "cerveaux/experience3_7")

# Cerveau 8

def sauvegarde_exp3_8() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(20)])
    C2 = Couche(ordre = 20, Adn = g, couche = [Neurone(ordre = 20, Adn = g) for i in range(30)])
    C3 = Couche(ordre = 30, Adn = g, couche = [Neurone(ordre =30, Adn = g) for i in range(10)])
    C4 = Couche(ordre = 10, Adn = g, couche = [Neurone(ordre = 10, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2, C3, C4])
    sv_cerveau(B, "cerveaux/experience3_8")

# Cerveau 9

def sauvegarde_exp3_9() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(20)])
    C2 = Couche(ordre = 20, Adn = g, couche = [Neurone(ordre = 20, Adn = g) for i in range(30)])
    C3 = Couche(ordre = 30, Adn = g, couche = [Neurone(ordre =30, Adn = g) for i in range(50)])
    C4 = Couche(ordre = 50, Adn = g, couche = [Neurone(ordre = 50, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2, C3, C4])
    sv_cerveau(B, "cerveaux/experience3_9")

# Cerveau 10

def sauvegarde_exp3_10() :
    C1 = Couche(ordre = 1, Adn = g, couche = [Neurone(ordre = 1, Adn = g) for i in range(20)])
    C2 = Couche(ordre = 20, Adn = g, couche = [Neurone(ordre = 20, Adn = g) for i in range(30)])
    C3 = Couche(ordre = 30, Adn = g, couche = [Neurone(ordre =30, Adn = g) for i in range(150)])
    C4 = Couche(ordre = 150, Adn = g, couche = [Neurone(ordre = 150, Adn = g)])
    B = Cerveau(Adn = g, reseau = [C1, C2, C3, C4])
    sv_cerveau(B, "cerveaux/experience3_10")

def experience3(z) :
    lx = [x/200 for x in range(201)]
    ly = [z(x) for x in lx]
    py.plot(lx, ly, label = "Courbe réelle")
    for k in {1,4,5,7,8,10} :
        R = Cerveau()
        s = str(k)
        R.generer("cerveaux/experience3_"+s, g)
        modele = [(np.array([x/(50)]),np.array([z(x/(50))])) for x in range(51)]
        for i in range(2000) :
            if i% 100 == 0 :
                print(k,i)
            apprentissage_gradient(R, modele, 0.5)
        ly = [R.influx_cerveau(np.array([x])) for x in lx]
        py.plot(lx,ly, label = s)
    py.legend(loc='best')
    py.show()
    
def experience3_carre() :
    def z(x) :
        return x**2
    experience3(z)

def experience3_sinus() :
    def z(x) :
        return 1 - (sin(4*x))**2
    experience3(z)
    
def experience3_exp() :
    def z(x) :
        return (exp(x) - 1)/(exp(1.) - 1)
    experience3(z)

#==============================================================================
# Test 4 : test 1 avec insistance
#==============================================================================
    
def experience4(z) :
    lx = [x/200 for x in range(201)]
    ly = [z(x) for x in lx]
    py.plot(lx, ly, label = "Courbe réelle")
    for k in range(1,2,2) :
        R = Cerveau()
        R.generer("cerveaux/experience1", g)
        modele = [(np.array([x/(5* k)]),np.array([z(x/(5* k))])) for x in range(5*k + 1)]
        for i in range(100) :
            if i% 100 == 0 :
                print(k,i)
            apprentissage_gradient(R, modele, 0.5)
        ly = [R.influx_cerveau(np.array([x])) for x in lx]
        s = str(k)
        py.plot(lx,ly, label = s)
    py.legend(loc='best')
    py.show()
    
def experience4_carre() :
    def z(x) :
        return x**2
    experience4(z)

def experience4_sinus() :
    def z(x) :
        return 1 - (sin(4*x))**2
    experience4(z)
    
def experience4_exp() :
    def z(x) :
        return (exp(x) - 1)/(exp(1.) - 1)
    experience4(z)