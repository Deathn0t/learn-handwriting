from PIL import Image
from PIL import ImageFilter
import glob, os
import pickle
import numpy as np
import sqlite3 as sq
from neurone_v3 import *
from multicouche_v3 import *

def lissage(pth):
    for infile in glob.glob(pth):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        im1 = im.filter(ImageFilter.DETAIL)
        im1.save(file+"_c"+ext, "BMP")

#lissage("enveloppes/*.bmp")

def trait(pth):
    for infile in glob.glob(pth):
        f, ext = os.path.splitext(infile)
        im = Image.open(infile)
        pixim = im.load()
        w, h = im.size
        xt = 395
        dbx = 1266
        yt = 101
        dby = 940
        tmp = Image.new("RGB", (xt, yt), "white")
        pixtmp = tmp.load()
        for y in range(dby,dby+yt):
            for x in range(dbx,dbx+xt):
                pixtmp[x-dbx, y-dby] = pixim[x, y]
        tmp.save(f+"_r"+ext, "BMP")

#trait("enveloppes/*_c.bmp")

def fichier_image_array():
    tmp = "D:/Users/Death/Pictures/donnees/"
    m = []
    for dossier in range(10):
        pth = tmp+str(dossier)+"/*.bmp"
        for infile in glob.glob(pth):
            f, ext = os.path.splitext(infile)
            im = Image.open(infile)
            x = list(im.getdata())
            for i in range(len(x)):
                x[i] = 1.-((x[i][0]+x[i][1]+x[i][2])/3./255.)
            x = np.array(x)
            m.append( (x, np.array([1. if (i == dossier) else 0. for i in range(10)]) ) )
    dest = tmp + str(len(m)//10)
    filehandler = open(dest, "wb")
    pickle.dump(m, filehandler)
    filehandler.close()

def image_array(loc, save ="data/Env1/2"):
    im = Image.open(loc)
    x = list(im.getdata())
    for i in range(len(x)):
        x[i] = 1.-((x[i][0]+x[i][1]+x[i][2])/3./255.)
    x = np.array(x)
    filehandler = open(save,"wb")
    pickle.dump(x,filehandler)
    filehandler.close()


def get_image(pth = "enveloppes/0.bmp"):
    image_array(pth)
    f = open("data/Env1/2",'rb')
    object_file = pickle.load(f)
    f.close()
    return object_file

def quelle_ville(cp):
    conn = sq.connect("codes_postaux.sql")
    c = conn.cursor()
    c.execute("""SELECT nom FROM villes WHERE cp=?""", (str(cp),))
    r = c.fetchall()
    c.close()
    conn.close()
    return r