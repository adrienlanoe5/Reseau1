import numpy as np
from mpmath.math2 import math_sqrt
#from scipy.special import expit
#np.seterr(all='raise')
import tkiteasy as tk

#np.seterr(all='ignore') #rajouté pour éviter des erreurs liées aux trop grandes valeurs : exp(-1000000000)=0

#X_train : liste à 2 niveaux
#1er niveau : une image
#2ème : plusieurs listes correspondant à chacune à un niveau de l'image
#Y_train : labels

class reseau_neurones():
    def __init__(self, liste_neurones, param_fonc_activation,taux_apprentissage):
        self.nb_neurones =liste_neurones
        self.nb_couches=len(self.nb_neurones) #ensemble des couches cachées et la derniere
        self.liste_poids= self.initialisation_poids()
        self.reussite=0
        self.defaite=0
        self.n=taux_apprentissage
        self.param=param_fonc_activation
        self.max_norme=0.5

    def initialisation_poids(self):
        liste=[]
        mat_1=self.tirage(self.nb_neurones[0],28*28+1)
        liste.append(mat_1)
        for i in range(1,self.nb_couches):
            mat=self.tirage(self.nb_neurones[i],self.nb_neurones[i-1])
            liste.append(mat)
        return liste

    def tirage(self, l,c):
        mat=np.random.uniform(-0.1,0.1,(l,c))
        return np.reshape(mat,(l,c))

    def normalisation_image(self,image):
        for i in range(len(image)):
            image[i] = image[i] / 255
        image = np.append(image, [1])
        return image

    def reset(self):
        self.reussite=0
        self.defaite=0

    def test(self,image,label_image):
        # forward propagation
        resultat_couche=np.reshape(self.normalisation_image(image),(28*28+1,1))
        for i in range(self.nb_couches):
            resultat_couche=np.matmul(self.liste_poids[i],resultat_couche)
            resultat_couche=self.fonction_activation(resultat_couche)

        vect_resultat=np.reshape(self.softmax(resultat_couche),(1,10))
        rang_resultat=np.argmax(vect_resultat[0])

        #performance
        self.performance(rang_resultat, label_image)

    def apprentissage(self,image,label_image):
        #forward propagation
        self.archi_resultats =[]
        self.archi_erreurs={}
        resultat_couche=np.reshape(self.normalisation_image(image),(28*28+1,1))
        for i in range(self.nb_couches):
            resultat_couche=np.matmul(self.liste_poids[i],resultat_couche)
            resultat_couche=self.fonction_activation(resultat_couche)
            self.archi_resultats.append(resultat_couche)


        vect_resultat=np.reshape(self.softmax(resultat_couche),(1,10))
        rang_resultat=np.argmax(vect_resultat[0])


        #performance
        self.performance(rang_resultat,label_image)

        # backward propagation
        # derniere couche
        vect_erreur=self.erreur_derniere_couche(vect_resultat,rang_resultat)
        vect_erreur=self.clipping_gradient(vect_erreur)
        self.archi_erreurs[self.nb_couches-1]=vect_erreur
        self.maj_poids(self.nb_couches-1,vect_erreur)

        #couches précédentes
        for i in range(self.nb_couches-2,-1,-1):
            vect_erreur=self.calcul_erreur(i)
            vect_erreur = self.clipping_gradient(vect_erreur)
            self.archi_erreurs[i]=vect_erreur
            self.maj_poids(i,vect_erreur)

    def clipping_gradient(self,vect):
        dim=np.shape(vect)
        sum=0
        for i in range (dim[0]):
            sum= sum+ float(vect[i])*float(vect[i])
        norme=math_sqrt(sum)
        if self.max_norme<norme:
           for k in range (dim[0]):
               nb=self.max_norme/norme
               vect[k]=nb*vect[k]

        return vect


    def erreur_derniere_couche(self,vect_resultat,rang_resultat):
        vect_erreur=[]
        for i in range(10):
            if i!=rang_resultat:
                erreur=-vect_resultat[0][i]
            else:
                erreur=1-vect_resultat[0][i]
            vect_erreur.append(erreur)
        vect_erreur=np.array(vect_erreur,dtype=object)
        return np.reshape(vect_erreur,(10,1))


    def calcul_erreur(self,i): #l'erreur serait dans cette fonction
        #formule erreur :
        # dérivée fonction d'activation avec en valeur la valeur de la fonction d'activation du poids
        # x la somme des erreurs pondérées de la couche i+1 par les poids des neurones de la couche i+1
        dim=np.shape(self.archi_erreurs[i+1])
        vect_trans_erreur_couche_suivante=np.reshape(self.archi_erreurs[i+1],(dim[1],dim[0]))
        vect=np.matmul(vect_trans_erreur_couche_suivante,self.liste_poids[i+1])
        dim =np.shape(vect)
        vect=np.reshape(vect,(dim[1],dim[0]))

        #vecteur des dérivées
        vect_res_couche_i=self.archi_resultats[i]
        dim=np.shape(vect_res_couche_i)
        vect_derivees=[]

        for k in range(dim[0]):
            nb=self.derivee_fonction_activation(vect_res_couche_i[k])
            vect_derivees.append(nb)
        vect_derivees=np.array(vect_derivees)

        #calcul final
        vect_erreur=self.produit_coordonnees(vect,vect_derivees)
        vect_erreur=np.reshape(vect_erreur,(dim[0],1))
        return vect_erreur

    def maj_poids(self,i,erreur):
        #calculs préliminaires
        mat_valeurs_neurones_erreur=self.produit_coordonnees(self.archi_resultats[i],erreur)
        dim=np.shape(mat_valeurs_neurones_erreur)
        mat=self.n*mat_valeurs_neurones_erreur

        #mise à la dimension correcte
        dim_voulue=np.shape(self.liste_poids[i])
        vect=[]
        for k in range (dim[0]):
            vect.append([mat[k] for j in range(dim_voulue[1])])
        mat_finale=np.array(vect)

        #calcul final
        new_mat_poids= self.liste_poids[i] + mat_finale
        self.liste_poids[i]=new_mat_poids
        #formule de la maj des poids :
        #new_poids= poids + learning rate x valeur neuronne couche i x erreur

    def produit_coordonnees(self,a,b):
        dim=np.shape(a)
        new_vect=[]
        for i in range(dim[0]):
            new_vect.append(np.matmul(a[i],b[i]))
        return np.array(new_vect)

    def fonction_activation(self,vect):
        if self.param=="sigmoide":
            return 1/(1 + np.exp(vect))
            # return expit(vect)
        elif self.param=="tangente hyperbolique":
            return (np.exp(vect)-np.exp(-vect))/(np.exp(vect)+np.exp(-vect))
        elif self.param=="selu":
            alpha = 1.67326324
            scale = 1.05070098
            for i in range(len(vect)):
                if vect[i] > 0:
                    vect[i]=scale*vect[i]
                else:
                    vect[i]=scale * alpha * (np.exp(vect[i]) - 1)
            return vect

        else:   # param=="tangente":
            return np.tan(vect)


    def derivee_fonction_activation(self, x):
        if self.param=="sigmoide":
            return np.exp(-x) / ((1 + np.exp(-x))**2)
        elif self.param=="tangente hyperbolique":
            return 1+((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2
        elif self.param=="selu":
            alpha = 1.67326324
            scale = 1.05070098
            for i in range(len(x)):
                if x[i] > 0:
                    x[i]=scale
                else:
                    x[i]=scale * alpha * np.exp(x[i])
            return x
        else:
            return 1+np.tan(x)**2

    def softmax(self,v):
        if v.ndim==1:
            v=v.reshape(1,-1)
        return np.exp(v) / np.sum(np.exp(v), axis=0,keepdims=True)


    def performance(self,label_pred, label_image):
            if label_pred != label_image :
                self.defaite += 1
            else:
                self.reussite += 1

    def taux_reussite(self):
        return self.reussite / (self.reussite + self.defaite)

    def prediction_dessin(self,image): #pour le test d'image dessinée à la main
        #forward propagation
        resultat_couche=np.reshape(image,(28*28+1,1))
        for i in range(self.nb_couches):
            resultat_couche=np.matmul(self.liste_poids[i],resultat_couche)
            resultat_couche=self.fonction_activation(resultat_couche)

        vect_resultat=np.reshape(self.softmax(resultat_couche),(1,10))
        rang_resultat=np.argmax(vect_resultat[0])

        return rang_resultat


# MNIST Dataset

import struct
from array import array

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# Set file paths based on added MNIST Datasets

training_images_filepath = 'Reseaudeneurones/archive/t10k-images.idx3-ubyte'
training_labels_filepath = 'Reseaudeneurones/archive/t10k-labels.idx1-ubyte'
test_images_filepath = 'Reseaudeneurones/archive/train-images.idx3-ubyte'
test_labels_filepath = 'Reseaudeneurones/archive/train-labels.idx1-ubyte'

training_images_filepath_dataform = 'données dentrainement formes/Data forme'


# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#Commandes phase de test
def declenchement(taux_apprentissage,type_fonction_acti,liste_neurones):
    Neurone = reseau_neurones(liste_neurones, type_fonction_acti,taux_apprentissage)
    # phase apprentissage
    for i in range(len(x_train)):
        new_image = np.ravel(x_train[i])
        Neurone.apprentissage(new_image, y_train[i])
    Neurone.reset()
    # phase de tests
    for i in range(len(x_test)):
        new_image = np.ravel(x_test[i])
        Neurone.test(new_image, y_test[i])
    return Neurone.taux_reussite()

def boucle(liste,objet):
    resultat=[]
    for i in range(len(liste)):
        if objet=="taux_apprentissage":
            resultat.append(declenchement(liste[i],"sigmoide",[2,2,6,10]))
        elif objet=="fonction_activation":
            resultat.append(declenchement(0.03,liste[i],[2,2,6,10]))
        else:
            resultat.append(declenchement(0.03, "sigmoide", liste[i]))
    print(objet)
    print(liste)
    print(resultat)
    print("______")

taux_apprentissage=[0.01, 0.03, 0.06, 0.25, 0.5, 0.75, 1]
type_fonction_acti=["sigmoide","tangente hyperbolique","tangente","selu"]
liste_neurones=[]
liste_couches=[]

#boucle(taux_apprentissage,"taux_apprentissage")
#boucle(type_fonction_acti,"fonction_activation")
#boucle(liste_neurones,"neurones")
#boucle(liste_couches,"couches")

def activer_Neurone(x_train, y_train,x_test,y_test):
    #commandes mise au point
    liste=[5,8,6,10]
    Neurone = reseau_neurones(liste, "selu", 0.03)
    #phase apprentissage
    for i in range(len(x_train)):
        new_image = np.ravel(x_train[i])
        Neurone.apprentissage(new_image, y_train[i])
    print(Neurone.taux_reussite())
    Neurone.reset()
    # phase de tests
    for i in range(len(x_test)):
        new_image = np.ravel(x_test[i])
        Neurone.test(new_image, y_test[i])
    print(Neurone.taux_reussite())



#interface _image
def interface_image():
    cote = 28 * 20
    hplus = 100
    g = tk.ouvrirFenetre(x=cote, y=cote + hplus)
    g.dessinerRectangle(x=0, y=0, l=cote, h=hplus, col='yellow')
    g.dessinerRectangle(x=0, y=hplus, l=cote, h=cote, col='white')
    txt = g.afficherTexte('cliquer pour dessiner', cote / 2, hplus / 2 + 1, col='black', sizefont=20)

    g.attendreClic() #attend que l'utilisateur clique
    g.changerTexte(txt,'dessin en cours')
    grande_matrice = [[0 for i in range(cote)] for j in range(cote)]  # matrice contenant l'ensemble des pixels dessinés par l'utilisateur : 1 si récupéré, 0 sinon
    while g.recupererClic()==None: #récupère les positions avant le clic de fin
        x, y = g.recupererPosition().x, g.recupererPosition().y
        grande_matrice[y-hplus][x]=1
        g.changerPixel(x, y, 'black')
        #colorie aussi le voisinnage
        g.changerPixel(x-1,y,'black') #gauche
        grande_matrice[y-hplus][x-1] = 1
        g.changerPixel(x+1, y, 'black') #droite
        grande_matrice[y-hplus][x-1] = 1
        g.changerPixel(x, y-1, 'black') #bas
        grande_matrice[y-hplus-1][x] = 1
        g.changerPixel(x, y+1, 'black') #haut
        grande_matrice[y-hplus+1][x] = 1

    matrice=[[0 for i in range(28)] for j in range(28)]
    for x in range(28):
        for y in range(28):
            # récupération de l'espace associé au pixel de la matrice
            lignes_matrice_pixel=grande_matrice[(cote//28)*x:(cote//28)*(x+1)]
            matrice_pixel=[]
            for i in range(cote//28):
                matrice_pixel.append(lignes_matrice_pixel[i][(cote // 28) * y: (cote // 28) * (y + 1)])
            #calcul de la coloration moyenne de la case
            a=np.sum(matrice_pixel)/(cote//28)**2
            #matrice[x][y]+=a*255 #annule la normalisation de l'image
    image=np.ravel(matrice)
    g.attendreClic()
    g.fermerFenetre()
    Neurone = reseau_neurones([2,2,6,10], "sigmoide", 0.03)
    #phase apprentissage
    for i in range(len(x_train)):
        new_image = np.ravel(x_train[i])
        Neurone.apprentissage(new_image, y_train[i])
    print(Neurone.taux_reussite())
    Neurone.reset()
    res=Neurone.prediction_dessin(image)
    print(res)

#Formes géométriques
# A faire :
#- séparer en 2 listes pour phase d'apprentissage et phase de test (voir discord)
#-fragmenter les images en 28x28
#- enlever les couleurs et les mettre en nuances de gris

import os
from PIL import Image

def ChargementBase(dossier):
    images = []
    labels = []
    noms_images = []

    # Parcours de tous les fichiers du dossier
    for fichier in os.listdir(dossier):
        chemin_complet = os.path.join(dossier, fichier)

        # Vérifier si c'est bien une image
        if fichier.endswith((".jpg", ".png", ".jpeg")):
            # Chargement et transformation de l'image
            image = Image.open(chemin_complet)
            image = image.resize((28, 28))
            image = image.convert("L")  # Convertir en niveaux de gris
            image_array = np.asarray(image).flatten()  # Aplatir en 1D

            # Définition du label basé sur le nom de l’image
            if "circle" in fichier.lower():
                label = 1
            elif "square" in fichier.lower():
                label = 2
            elif "triangle" in fichier.lower():
                label = 3
            elif "kite" in fichier.lower():
                label = 4
            elif "parallelogram" in fichier.lower():
                label = 5
            elif "rectangle" in fichier.lower():
                label = 6
            elif "rhombus" in fichier.lower():
                label = 7
            elif "trapezoid" in fichier.lower():
                label = 8
            else:
                label = 0  # Label par défaut si inconnu

            # Stocker l'image, son label et son nom
            images.append(image_array)
            labels.append(label)
            noms_images.append(fichier)

    # Conversion en tableaux NumPy
    images = np.array(images, dtype=np.float32)  # Optionnel : Normalisation possible
    labels = np.array(labels, dtype=np.int32)

    #combinaison = np.column_stack((images, labels))
    #np.random.shuffle(combinaison)
    #images = combinaison[:, 0]
    #labels = combinaison[:, 1]

    images_train = images[:3000]
    images_test = images[3000:]
    labels_train = labels[:3000]
    labels_tests = labels[3000:]

    print(f"Base chargée avec {len(images)} images")
    print(f"Noms des premières images : {noms_images[:5]}")
    print(f"Labels des premières images : {labels[:5]}")

    return (images_train,labels_train),(images_test,labels_tests)


images_filepath="données d'entrainement formes/Data forme"
(images_train,labels_train),(images_test,labels_tests)= ChargementBase(images_filepath)


#interface_image()
#activer_Neurone(x_train, y_train,x_test,y_test)
#activer_Neurone(images_train,labels_train, images_test,labels_tests)