from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from neurone_v3 import *
from multicouche_v3 import *


C1 = Couche("C1", 784, Sigmoid, [Neurone(str(i), 784, Sigmoid) for i in range(15)])
C2 = Couche("C2", 15, Sigmoid, [Neurone(str(i), 15, Sigmoid) for i in range(10)])
Br = Cerveau("Br", Sigmoid, [C1, C2])

fichier = "cerveaux/test_series_c"

sv_cerveau(Br, fichier) #Sauvegarde pour pouvoir fixer l'initialisation du cerveau

m1 = gt_modele('modele/20elts')

s1 = [i for i in range(len(m1)) for j in range(10)]
s2 = [j for i in range(10) for j in range(len(m1))]

series = [list(np.random.randint(0,20,20)) for i in range(10)] #series de valeurs, ATTENTION, si les sous listes sont de tailles differentes NON SENS

cerveaux = [Cerveau() for i in range(len(series))]
for c in cerveaux:
    c.generer(fichier, Sigmoid)


fig = plt.figure()

x = [i for i in range(len(cerveaux))]
y = [i+1 for i in range(len(cerveaux))]
data = [0 for i in range(len(cerveaux))]
rects = plt.bar(x, data)
plt.ylim(0, 100)
plt.xlim(-0.2,10)

def animate(i):
    for k,rect in enumerate(rects):
        gradLearn(cerveaux[k], m1, series[k], 1.)
        data[k] = bonnes_reponses(cerveaux[k],m1)
        rect.set_height(data[k])
        series[k] = list(np.random.randint(0,20,20))
    return rects

anim = animation.FuncAnimation(fig, animate, frames=1000, interval=10)
plt.show()