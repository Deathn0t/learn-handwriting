import pylab as py
from multicouche_v3 import *
from neurone_v3 import *
from img_trt import get_image
from time import *

def arrondi(x,dec):
    return int((x*(10.**dec))+0.5)/(10.**dec)

def chrono(f):
    t1 = time()
    f()
    t2 = time()
    return arrondi(t2-t1,2)

C1 = Couche("C1", 784, Sigmoid, [Neurone(str(i), 784, Sigmoid) for i in range(15)])
C2 = Couche("C2", 15, Sigmoid, [Neurone(str(i), 15, Sigmoid) for i in range(10)])
Br = Cerveau("Br", Sigmoid, [C1, C2])
Br1 = Cerveau()
Br2 = Cerveau()
Br1.generer("cerveaux/encours1",Sigmoid)
Br2.generer("cerveaux/encours2",Sigmoid)

m1 = gt_modele('modele/50000')
m2 = m1[:200]
#m2 = m1[:20000]
#m3 = m1[20000:]
f = lambda : get_image("C:/Users/Death/Dropbox/TIPE/Programmes Python/enveloppes/0.bmp")
inf = lambda im: maxL(Br.influx_cerveau(im))
 
def go(cerveau, modele, itr, alpha):
    for i in range(itr):
        serie = list(np.random.randint(0,len(modele), len(modele)//1000))
        gradLearn(cerveau, modele, serie, alpha)
        
def test():
    i = 0
    alpha = 1.
    lx = [i]
    b1, b2 = bonnes_reponses(Br,m2), bonnes_reponses(Br,m2, arrondiS)
    ly1 = [b1]
    ly2 = [b2]
    ly3 = [cout_totale(Br,m2)]
    while (b1 < 100.):
        if (i % 10) == 0:
            print(i, "=", b1, "=", b2)
        if (b1 < 95):
            serie = list(np.random.randint(0,len(m2), len(m2)//10))
            gradLearn(Br, m2, serie, alpha)
        else:
            serie = reponses(Br,m2)
            gradLearn(Br, m2, serie, alpha)
        i += 1
        lx.append(i)
        b1, b2 = bonnes_reponses(Br,m2), bonnes_reponses(Br,m2, arrondiS)
        ly1.append(b1)
        ly2.append(b2)
        c = cout_totale(Br,m2)
        ly3.append(c)
    py.plot(lx,ly1, label ="y1")
    py.plot(lx,ly2, label ="y2")
    py.plot(lx,ly3, label ="y3")
    py.legend()
    py.show()

def count(m):
    l = [0 for i in range(10)]
    for a, b in m:
        #print(l, b)
        pos, r = maxL(b)
        l[pos] = l[pos] + 1
    return l

def app(cerveau, modele, alpha):
    i = 0
    b2 = bonnes_reponses(cerveau, modele, arrondiM)
    while (b2 < 95.):
        if ( (i % 10) == 0 ):
            print(i, "=", b2)
            b2 = bonnes_reponses(cerveau, modele, arrondiM)
        if ( b2 < 85. ):
            serie = list(np.random.randint(0,len(modele), 50))
            gradLearn(Br, modele, serie, alpha)
        else:
            serie = reponses(cerveau,modele)
            subserie = list(np.random.randint(0,len(serie), 50))
            serie = [serie[j] for j in subserie]
            gradLearn(cerveau, modele, serie, alpha)
        i += 1
        
def simulation(cerveaux, modele, ech):
    bonD = 0
    bonC = 0
    t = len(cerveaux)
    mx = len(modele)
    for i in range(ech):
        elts = np.random.randint(0,mx,5)
        cpTh, cpCs = [], [[] for i in range(t)]
        for j in elts:
            cpTh.append(maxL(modele[j][1])[0])
            for k,c in enumerate(cerveaux):
                cpCs[k].append(maxL(c.influx_cerveau(modele[j][0]))[0])
            k = 0
            if len(cpTh) == 2:
                while (k < t) and not(cpCs[k] == cpTh):
                    k = k + 1
                if (k < t):
                    bonD += 1
        while (k < t) and not(cpCs[k] == cpTh):
            k = k + 1
        if (k < t):
            bonC += 1
    return float(bonD)/float(ech), float(bonC)/float(ech)
        
    

if __name__ == '__main__':
    pass