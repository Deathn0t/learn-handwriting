from img_trt import *
from multicouche_v3 import *
from neurone_v3 import *

Br = Cerveau()
Br.generer("cerveaux/f200", Sigmoid)
f = lambda x: get_image("C:/Users/Death/Dropbox/TIPE/Programmes Python/enveloppes/"
                       +str(x)+".bmp")
inf = lambda im: maxL(Br.influx_cerveau(im))

def code():
    cp = ""
    for i in range(5):
        im = f(i)
        r = inf(im)
        cp += str(r[0])
    return int(cp)

def Q():
    try:
        return quelle_ville(code()), code()
    except:
        return "Mauvais Code Postal", code()