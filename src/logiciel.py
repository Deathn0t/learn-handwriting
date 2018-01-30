from Tkinter import *
from neurone_v3 import *

CERVEAUX = []
cmp_cerveaux = 0
cmp_couches = 0
cmp_neurones = 0

def renvoie_obj(nom, l):
    for c in l:
        if str(c) == nom:
            return c

def cercle(canvas, x, y, r, **kwargs):
    return canvas.create_oval(x-r, y-r, x+r, y+r,**kwargs)

def afficher_nom(canvas, x, y, texte):
    return canvas.create_text(x, y, text=texte, font="Arial 12 italic", fill="white")

def dessiner_neurone(canvas, nom, x, y ,r, **kwargs):
    cercle(canvas, x, y ,r, **kwargs)
    afficher_nom(canvas, x, y, nom)

def click(event):
    print(event.x, event.y)

def alert():
    print("Pas de fonction attribue")

def active_cerveau(t):
    maj_lb_couches()
    affichage_simple()

def active_couche(t):
    maj_lb_neurones()

def test():
    pass

def creer_cerveau():
    global  cmp_cerveaux
    CERVEAUX.append(Cerveau("C"+str(cmp_cerveaux), Perceptron))
    list_cerveaux.insert(END, "C"+str(cmp_cerveaux))
    cmp_cerveaux += 1

def creer_couche():
    global cmp_couches
    nom_cerveau = list_cerveaux.get(ACTIVE)
    cerv = renvoie_obj(nom_cerveau, CERVEAUX)
    cerv.ajout_couche()
    cerv.couches[-1].nom_couche = "c"+str(cmp_couches)
    cmp_couches += 1
    maj_lb_couches()

def creer_neurone():
    global cmp_neurones
    nom_cerveau = list_cerveaux.get(ACTIVE)
    nom_couche = list_couches.get(ACTIVE)
    cerv = renvoie_obj(nom_cerveau, CERVEAUX)
    cou = renvoie_obj(nom_couche, cerv.couches)
    cerv.ajout_neurone(cou)
    cou.couche[-1].nom_neurone = "N"+str(cmp_neurones)
    cmp_neurones += 1
    maj_lb_neurones()

def supprimer_cerveau(event =None):
    cerv = renvoie_obj(list_cerveaux.get(ACTIVE), CERVEAUX)
    CERVEAUX.remove(cerv)
    list_cerveaux.delete(ANCHOR)

def supprimer_couche(event =None):
    cerv = renvoie_obj(list_cerveaux.get(ACTIVE), CERVEAUX)
    cou = renvoie_obj(list_couches.get(ACTIVE), cerv.couches)
    cerv.couches.remove(cou)
    list_couches.delete(ANCHOR)

def supprimer_neurone(event = None):
    cerv = renvoie_obj(list_cerveaux.get(ACTIVE), CERVEAUX)
    cou = renvoie_obj(list_couches.get(ACTIVE), cerv.couches)
    neu = renvoie_obj(list_neurones.get(ACTIVE), cou.couche)
    cou.couche.remove(neu)
    list_neurones.delete(ANCHOR)

def maj_lb_couches():
    list_couches.delete(0, END)
    nom_cerveau = list_cerveaux.get(ACTIVE)
    cerv = renvoie_obj(nom_cerveau, CERVEAUX)
    for couche in cerv.couches:
        list_couches.insert(END, str(couche))
    maj_lb_neurones()

def maj_lb_neurones():
    list_neurones.delete(0, END)
    nom_cerveau = list_cerveaux.get(ACTIVE)
    nom_couche = list_couches.get(ACTIVE)
    cerv = renvoie_obj(nom_cerveau, CERVEAUX)
    couche = renvoie_obj(nom_couche, cerv.couches)
    if couche != None:
        for neurone in couche.couche:
            list_neurones.insert(END, str(neurone))

def dessiner_cerveau(cerv, activite =None):
    rayon = 10+ 5
    pas = 2*rayon
    color1 = color2 = 'black'
    for i in range(len(cerv.couches)):
        for j in range(len(cerv.couches[i].couche)):
            if not(activite == None):
                color1 = 'red' if activite[i+1][j] == True else 'black'
            neu = cerv.couches[i].couche[j]
            dessiner_neurone(canvas, str(neu), i*(5*pas)+(15*rayon), j*(2*pas)+(2*rayon), rayon, fill =color1)
            for k in range(cerv.couches[i].ordre):
                color2 = 'red' if not(activite == None) and activite[i][k] == True else 'black'
                if i == 0:
                    canvas.create_line((4*rayon), k*(2*pas)+(2*rayon),
                                   (15*rayon), j*(2*pas)+(2*rayon), fill =color2)
                    e1 = Entry(canvas, width =5)
                    canvas.create_window((4*rayon), k*(2*pas)+(2*rayon), window=e1)
                else:
                    canvas.create_line((i-1)*(5*pas)+(15*rayon), k*(2*pas)+(2*rayon),
                                   (i)*(5*pas)+(15*rayon), j*(2*pas)+(2*rayon), fill =color2)

def affichage_simple():
    restaurer()
    nom_cerv = list_cerveaux.get(ACTIVE)
    if nom_cerv == '':
        print("Pas de Cerveaux")
    else:
        cerv = renvoie_obj(nom_cerv, CERVEAUX)
        dessiner_cerveau(cerv)

def affichage_stimulation(entree =[1,1,1,1]):
    restaurer()
    nom_cerv = list_cerveaux.get(ACTIVE)
    if nom_cerv == '':
        print("Pas de Cerveaux")
    else:
        cerv = renvoie_obj(nom_cerv, CERVEAUX)
        activite = influx_entree(entree)
        dessiner_cerveau(cerv, activite)

def influx_entree(entree):
    sortie = entree[:]
    activite = [[True if (i >= 0.5) else False for i in sortie]]
    nom_cerv = list_cerveaux.get(ACTIVE)
    cerv = renvoie_obj(nom_cerv, CERVEAUX)
    for couche in cerv.couches:
        sortie = couche.influx_couche(sortie)
        supp = [True if (i >= 0.5) else False for i in sortie]
        activite.append(supp[:])
    return activite


def restaurer():
    canvas.delete(ALL)

####MAIN####

fenetre = Tk()

canvas = Canvas(fenetre, bd =5, relief=RAISED, width=900, height=550, cursor="cross")
canvas.grid(row =1, column =1, columnspan =10)
canvas.bind("<Button-1>", click)

#vscale = DoubleVar()
#scale = Scale(fenetre, from_=1, to=20, orient=VERTICAL, length =500, variable=vscale)
#scale.grid(row =1, column =2)

fr_lcerveaux = Frame(fenetre)
scrol_lcerveaux = Scrollbar(fr_lcerveaux, orient=VERTICAL)
list_cerveaux = Listbox(fr_lcerveaux, yscrollcommand=scrol_lcerveaux.set, height=10)
list_cerveaux.bind("<Double-Button-1>", active_cerveau)
list_cerveaux.bind("<Delete>", supprimer_cerveau)
scrol_lcerveaux.config(command=list_cerveaux.yview)
scrol_lcerveaux.pack(side=RIGHT, fill=Y)
list_cerveaux.pack(side=LEFT, fill=BOTH, expand=1)
fr_lcerveaux.grid(row =1, column =11)

fr_lcouches = Frame(fenetre)
scrol_lcouches = Scrollbar(fr_lcouches, orient=VERTICAL)
list_couches = Listbox(fr_lcouches, yscrollcommand=scrol_lcouches.set, height=10)
list_couches.bind("<Double-Button-1>", active_couche)
list_couches.bind("<Delete>", supprimer_couche)
scrol_lcouches.config(command=list_couches.yview)
scrol_lcouches.pack(side=RIGHT, fill=Y)
list_couches.pack(side=LEFT, fill=BOTH, expand=1)
fr_lcouches.grid(row =1, column =12)

fr_lneurones = Frame(fenetre)
scrol_lneurones = Scrollbar(fr_lneurones, orient=VERTICAL)
list_neurones = Listbox(fr_lneurones, yscrollcommand=scrol_lneurones.set, height=10)
list_neurones.bind("<Delete>", supprimer_neurone)
scrol_lneurones.config(command=list_neurones.yview)
scrol_lneurones.pack(side=RIGHT, fill=Y)
list_neurones.pack(side=LEFT, fill=BOTH, expand=1)
fr_lneurones.grid(row =1, column =13)

menubar = Menu(fenetre)

menu1 = Menu(menubar, tearoff=0)
menu1.add_command(label="Creer", command=alert)
menu1.add_command(label="Editer", command=alert)
menu1.add_separator()
menu1.add_command(label="Quitter", command=fenetre.quit)
menubar.add_cascade(label="Fichier", menu=menu1)

b1 = Button(fenetre, text="RESTAURER", command=restaurer)
b1.grid(row =2, column =1)
b2 = Button(fenetre, text="AFFICHER CERVEAU", command=affichage_simple)
b2.grid(row =2, column =2)
b3 = Button(fenetre, text="STIMULATION", command=affichage_stimulation)
b3.grid(row =2, column =3)
b4 = Button(fenetre, text="AJOUTER CERVEAU", command=creer_cerveau)
b4.grid(row =2, column =8)
b5 = Button(fenetre, text="AJOUTER COUCHE", command=creer_couche)
b5.grid(row =2, column =9)
b6 = Button(fenetre, text="AJOUTER NEURONE", command=creer_neurone)
b6.grid(row =2, column =10)

b11 = Button(fenetre, text="SUPPRIMER CERVEAU", command=supprimer_cerveau)
b11.grid(row =2, column =11)
b12 = Button(fenetre, text="SUPPRIMER COUCHE", command=supprimer_couche)
b12.grid(row =2, column =12)
b13 = Button(fenetre, text="SUPPRIMER NEURONE", command=supprimer_neurone)
b13.grid(row =2, column =13)




fenetre.config(menu=menubar)

fenetre.mainloop()