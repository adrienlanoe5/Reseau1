#X_train : liste à 2 niveaux
#1er niveau : une image
#2ème : plusieurs listes correspondant à chacune à un niveau de l'image
#Y_train : labels

#liste choses à faire:
#- cross entropy loss

#Questions:
#- np.transpose marche sur les vecteurs colonne ?

import numpy as np

class reseau_neurones():
    def __init__(self, liste_neurones):
        self.nb_neurones =liste_neurones
        self.nb_couches=len(self.nb_neurones) #ensemble des couches cachées et la derniere
        self.liste_poids= self.initialisation_poids()
        self.reussite=0
        self.defaite=0
        self.n=0.03

    def initialisation_poids(self):
        liste=[]
        mat_1=np.zeros((self.nb_neurones[0],28*28+1))
        liste.append(mat_1)
        for i in range(1,self.nb_couches):
            mat=np.zeros((self.nb_neurones[i],self.nb_neurones[i-1]+1))
            liste.append(mat)
        return np.array(liste)

    def normalisation_image(self,image):
        for i in range(len(image)):
            image[i] = image[i] / 255
        return image

    def reset(self):
        self.reussite=0
        self.defaite=0

    def test(self,image,label_image):
        # forward propagation
        resultat_couche = self.normalisation_image(image)
        for i in range(self.nb_couches):
            resultat_couche = self.forward_propagation_produit_matriciel(i, resultat_couche)
            resultat_couche = self.fonction_activation(resultat_couche)

        resultat = self.softmax(resultat_couche)
        label_pred = str(np.argmax(resultat))

        # performance
        self.performance(label_pred, label_image)

    def apprentissage(self,image,label_image):
        #forward propagation
        self.archi_resultats =[]
        self.archi_erreurs=[]
        resultat_couche=np.reshape(self.normalisation_image(image),(28*28,1))
        for i in range(self.nb_couches):
            resultat_couche=self.forward_propagation_produit_matriciel(i,resultat_couche)
            resultat_couche=self.fonction_activation(resultat_couche)
            self.archi_resultats.append(resultat_couche)
        vect_resultat=np.reshape(self.softmax(resultat_couche),(1,10))
        rang_resultat=np.argmax(vect_resultat)
        label_pred=str(vect_resultat[rang_resultat])

        #performance
        self.performance(label_pred,label_image)

        # backward propagation
        # derniere couche
        vect_erreur=np.array(self.erreur_derniere_couche(vect_resultat,rang_resultat))
        self.archi_erreurs.append(vect_erreur)
        self.maj_poids(self.nb_couches,vect_erreur)

        #couches précédentes
        for i in range(self.nb_couches-1,0,-1):
            vect_erreur=self.calcul_erreur(i)
            self.archi_erreurs.append(vect_erreur)
            self.maj_poids(i,vect_erreur)



    def forward_propagation_produit_matriciel(self, couche, inputs):
        vect_resultat=np.matmul(self.liste_poids[couche],inputs)
        new_vect=np.append(vect_resultat, [1])
        return np.array(new_vect)

    def erreur_derniere_couche(self,vect_resultat,rang_resultat):
        dim=vect_resultat.shape()
        vect_erreur=[]
        for i in range(dim[1]):
            if i!=rang_resultat:
                erreur=-int(vect_erreur[i])
                vect_erreur.append(vect_erreur)
            else:
                vect_erreur.append(1-int(vect_erreur[i]))
        return np.array(vect_erreur)


    def calcul_erreur(self,i):
        #formule erreur :
        # dérivée fonction d'activation avec en valeur la valeur de la fonction d'activation du poids
        # x la somme des erreurs pondérées de la couche i+1 par les poids des neurones de la couche i+1
        dim=self.archi_erreurs[i+1].shape()
        vect_trans_erreur_couche_suivante=np.reshape(self.archi_erreurs[i+1],(dim[1],dim[0]))
        #vect_trans_erreur_couche_suivante = np.transpose(self.archi_erreurs[i + 1])
        vect=np.matmul(vect_trans_erreur_couche_suivante,self.liste_poids[i+1])
        del vect[-1]
        dim =vect.shape()
        vect=np.reshape(vect,(dim[1],dim[0]))

        #vecteur des dérivées
        vect_res_couche_i=self.archi_resultats[i]
        dim=vect_res_couche_i.shape()
        vect_derivees=[]
        for i in range(dim[0]):
            nb=self.derivee_fonction_activation(vect_res_couche_i[i])
            vect_derivees.append(nb)
        vect_derivees=np.array(vect_derivees)

        #calcul final
        vect_erreur=self.produit_coordonnees(vect, vect_derivees)
        return vect_erreur

    def produit_coordonnees(self,a,b):
        dim=a.shape()
        new_vect=[]
        for i in range(dim[0]):
            new_vect.append(np.matmul(a[i],b[i]))
        return np.array(new_vect)

    def maj_poids(self,i,erreur):
        #calculs préliminaires
        mat_valeurs_neurones_erreur=self.produit_coordonnees(self.archi_resultats[i],erreur)
        dim = mat_valeurs_neurones_erreur.shape()
        vect_learning_rate=np.array([self.n for k in range(dim[0])])
        dim=vect_learning_rate.shape()
        vect_learning_rate=np.reshape(vect_learning_rate,(dim[1],dim[0]) )
        mat=self.produit_coordonnees(vect_learning_rate,mat_valeurs_neurones_erreur)

        #mise à la dimension correcte
        dim_voulue=self.liste_poids[i].shape()
        vect=[]
        for i in range (dim[0]):
            vect.append([mat[i] for k in range(dim_voulue[1])])
        mat_finale=np.array(vect)

        #calcul final
        new_mat_poids= self.liste_poids[i] + mat_finale
        self.liste_poids[i]=np.array(new_mat_poids)
        #formule de la maj des poids :
        #new_poids= poids + learning rate x valeur neuronne couche i x erreur

    def fonction_activation(self,x):
        return 1/(1 + np.exp(-x)) #sigmoide

    def derivee_fonction_activation(self, x):
        return np.exp(-x) / ((1 + np.exp(-x))**2)

    def softmax(self,liste): #axis à tester
        return np.exp(liste) / np.sum(np.exp(liste), axis=0)

    def performance(self,label_pred, label_image):
            if label_pred != label_image :
                self.defaite += 1
            else:
                self.reussite += 1

    def taux_reussite(self):
        return self.reussite / (self.reussite + self.defaite)




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


# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


liste=[2,7,5,3,8,4,10]
Neurone=reseau_neurones(liste)
#phase apprentissage
for i in range (len(x_train)) :
    new_image=np.ravel(x_train[i])
    Neurone.apprentissage(new_image, y_train[i])

#print(Neurone.taux_reussite())
#Neurone.reset()

#phase de tests
for i in range (len(x_test)) :
    new_image=np.ravel(x_test[i])
    Neurone.test(new_image, y_test[i])
#print(Neurone.reussite,Neurone.defaite)

#print(Neurone.taux_reussite())