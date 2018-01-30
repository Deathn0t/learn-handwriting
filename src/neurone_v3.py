#Anaconda 2.7.8 (Python)

#from math import exp
import numpy as np
import pickle

###Boite A Outils###

def get_mot(texte, separateur ="."):
    phrase = ""
    for i,l in enumerate(texte):
        if (l == separateur):
            return phrase,texte[i+1:]
        #if '"' == l:
        #    continue
        phrase = phrase+l

def get_suite_mots(texte, serie, separateur ="."):
    l = []
    i = 0
    while separateur in texte:
        mot, texte = get_mot(texte, separateur)
        if i in serie:
            l.append(mot)
        i = i + 1
    return tuple(l)
        

####################

class Genome(object):
    """ Classe correspondant a la fonction d'activation, permettant de verifier
        que tous les neurones ont la meme fonction d'activation. """
    def __init__(self, nom_genome ="Heaviside",
                 activation =lambda x: 0 if (x < 0) else 1):
        self.nom_genome = nom_genome
        self.activation = activation

    def __str__(self):
        return self.nom_genome

class Neurone(object):
    """ Classe d'un Neurone. """
    def __init__(self, nom_neurone ="default", ordre =1, Adn =None):
        self.nom_neurone = nom_neurone
        self.Adn = Adn
        self.ordre = ordre
        self.activation = Adn.activation
        self.vect_w = np.array([np.random.random_sample()-np.random.random_sample() for i in range(self.ordre)])
        self.biais = np.random.random_sample()-np.random.random_sample()

    def __str__(self) :
        return self.nom_neurone

    def prod_scal(self, x, y):
        return np.dot(x,y)

    def influx(self, vect_x):
        ''' Calcule l'image d'un vecteur par un neurone. '''
        self.sortie_tmp = self.activation( self.prod_scal(vect_x, self.vect_w) + self.biais)
        return self.sortie_tmp

class Couche(object):
    """ Classe Couche."""
    def __init__(self, nom_couche ="defaut", ordre =1, Adn =None, couche =[]):
        self.nom_couche = nom_couche
        self.Adn = Adn
        self.ordre = ordre
        self.activation = Adn.activation
        self.couche = couche[:] # liste vide de neurones
        self.verifie_couche()

    def __str__(self):
        return self.nom_couche
        
    def verifie_couche(self):
        ''' Verifie que la couche initialisee par l'utilisateur est viable. '''
        for neur in self.couche:
                if (self.Adn != neur.Adn):
                    raise Warning ("Genome non respecte : "+str(self)+" -> "+str(neur))
                if (self.ordre != neur.ordre):
                    raise Warning ("Ordre non respecte : "+str(self)+" -> "+str(neur))

    def ajout_neurone(self, neurone =None):
        ''' Verifie que le neurone ajoute par l'utilisateur est viable. '''
        if neurone == None:
            self.couche.append(Neurone("N"+str(len(self.couche)), self.ordre, Adn =self.Adn))
        else:
            if neurone.Adn == self.Adn and neurone.ordre == self.ordre:
                self.couche.append(neurone)
            else:
                raise Warning ("Genome non respecte!")

    def influx_couche(self, vect_x):
        ''' Pour une couche [Ni] de neurones et un vecteur x, on renvoit la liste
            [Ni(x)]. '''
        sortie = []
        for neurone in self.couche:
            sortie.append(neurone.influx(vect_x))
        return np.array(sortie)
                

class Cerveau(object):
    """ Classe Cerveau. """
    def __init__(self, nom_cerveau ="default", Adn =None, reseau =[]):
        self.nom_cerveau = nom_cerveau
        self.Adn = Adn
        self.reseau = reseau[:] # liste vide de couches
        self.verifie_cerveau()

    def __str__(self):
        return self.nom_cerveau

    def structure(self):
        " strucutre = 'nom_cerveau.taille_reseau.taille_couche_entree.taille_couche1.taille_couche2...taille_coucheN.'"
        if (len(self.reseau) == 0):
            return "VIDE"
        struc = ""
        struc = struc + str(self) + "." + str(len(self.reseau)) + "." + str(self.reseau[0].ordre) + "."
        for couche in self.reseau:
            struc = struc + str(len(couche.couche)) + "."
        return struc
    
    def detruire(self):
        self.nom_cerveau = ''
        self.Adn = None
        self.reseau = []

    def generer(self, fichier, genome):
        f = open(fichier, 'r')
        struc = f.readline()
        self.construire(struc, genome)
        self.remplir(f)
                    
    def construire(self, structure, genome):
        " strucutre = 'nom_cerveau.taille_reseau.taille_couche_entree.taille_couche1.taille_couche2...taille_coucheN.'"
        self.nom_cerveau, struc = get_mot(structure)
        self.Adn = genome
        taille_cerveau, struc = get_mot(struc)
        taille_couche_entree, struc = get_mot(struc)
        for i in range(int(taille_cerveau)):
            if (i == 0):
                self.ajout_couche()
                self.reseau[0].ordre = int(taille_couche_entree)
                taille, struc = get_mot(struc)
                for j in range(int(taille)):
                    self.ajout_neurone(self.reseau[0])
            else:
                self.ajout_couche()
                taille, struc = get_mot(struc)
                for j in range(int(taille)):
                    self.ajout_neurone(self.reseau[-1])

    def remplir(self, f):
        for i in range(len(self.reseau)):
            for j in range(len(self.reseau[i].couche)):
                for k in range(len(self.reseau[i].couche[j].vect_w)):
                    s = float(f.readline())
                    self.reseau[i].couche[j].vect_w[k] = s
                self.reseau[i].couche[j].biais = float(f.readline())
        f.close()

    def verifie_cerveau(self) :
        for i in range(0, len(self.reseau)) :
            self.reseau[i].verifie_couche()
            if (self.Adn != self.reseau[i].Adn) :
                raise Warning ("Genome non respecte : "+str(self.reseau[i]))
            if (i>0) and (self.reseau[i].ordre != len(self.reseau[i-1].couche)) :
                raise Warning ("Ordre non respecte : "+str(self.reseau[i]))

    def influx_cerveau(self, vect_x):
        sortie = vect_x
        for couche in self.reseau:
            sortie = couche.influx_couche(sortie)
        return sortie
        
    def influx_app(self, entree):
        mat = [entree[:]]
        for cou in self.reseau:
            res = cou.influx_couche(mat[-1])
            mat.append(res[:])
        return mat[1:]

    def ajout_couche(self, couche =None):
        taille_cerveau = len(self.reseau)
        if (couche == None): #Ajout d'une couche avec parametres initialisees par defaut
            if (taille_cerveau == 0): #Ajout d'une couche dans le cas ou le cerveau est vide
                self.reseau.append(Couche(nom_couche ="c0", Adn = self.Adn))
            else: #Cas ou le cerveau est non vide, Ajout en queue
                self.reseau.append(Couche(nom_couche ="c"+str(len(self.reseau)),
                                           ordre =len(self.reseau[-1].couche),
                                           Adn =self.Adn))
        else: #Ajout d'une couche specifique
            if (couche.Adn == self.Adn): #On verifie la compatibilite du genome
                if (taille_cerveau == 0): #Si le cerveau est vide on ne verifie pas l'ordre
                    self.reseau.append(couche)
                elif (len(self.reseau[-1].couche) == couche.ordre): #Cerveau non vide, on verifie l'ordre
                    self.reseau.append(couche)
                else:
                    raise Warning ("Ordre non respecte!")
            else:
                raise Warning ("Genome non respecte!")

    def ajout_neurone(self, couche, neurone =None):
        cmp = 0
        while (cmp < len(self.reseau)-1) and (couche != self.reseau[cmp]):
            cmp += 1
        if cmp < len(self.reseau):
            couche.ajout_neurone(neurone)
        if cmp < len(self.reseau)-1:
            self.reseau[cmp+1].ordre = len(couche.couche)
            for neurone in self.reseau[cmp+1].couche:
                neurone.ordre = len(couche.couche)

####Genomes Usuels####
Perceptron = Genome()
Sigmoid = Genome("Sigmoid", lambda x: 1 / (1 + np.exp(-x)))

###Sauvegarde###

def sv_cerveau(cerveau, fichier):
    """
    Ne sauvegarde que les poids et biais
    :param cerveau: Cerveau a enregistrer
    :param fichier: Localisation et nom du fichier (Ex: Alpha/Omega/sauvegarde1
    """
    f = open(fichier,'w')
    f.write(cerveau.structure()+"\n")
    for couche in cerveau.reseau:
        for neurone in couche.couche:
            for w in neurone.vect_w:
                f.write(str(w)+"\n")
            f.write(str(neurone.biais)+"\n")
    f.close()

def sv_modele(fichier, modele):
    f = open(fichier, "wb")
    pickle.dump(modele, f)
    f.close()

def gt_modele(fichier):
    f = open(fichier, "rb")
    modele = pickle.load(f)
    f.close()
    return modele

###Main###
if __name__ == '__main__':
    nb_entrees = 4
    N1 = Neurone("N1", nb_entrees, Perceptron)
    N2 = Neurone("N2", nb_entrees, Perceptron)
    N3 = Neurone("N3", nb_entrees, Perceptron)
    N4 = Neurone("N4", nb_entrees, Perceptron)
    N5 = Neurone("N5", 4, Perceptron)
    N6 = Neurone("N6", 4, Perceptron)
    N7 = Neurone("N7", 2, Perceptron)
    N8 = Neurone("N8", 2, Perceptron)
    N9 = Neurone("N9", 2, Perceptron)
    C1 = Couche("C1", nb_entrees, Perceptron, [N1, N2, N3, N4])
    C2 = Couche("C2", 4, Perceptron, [N5, N6])
    C3 = Couche("C3", 2, Perceptron, [N7, N8, N9])
    Brain = Cerveau("Perceptron", Perceptron, [C1, C2, C3])
    sv_cerveau(Brain, "cerveaux/test")
    Br = Cerveau()
    Br.generer("cerveaux/test", Perceptron)
    x = np.array([1., 0., 0., 1.])


