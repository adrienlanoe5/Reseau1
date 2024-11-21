#X_train : liste à 2 niveaux
#1er niveau : une image
#2ème : plusieurs listes correspondant à chacune à un niveau de l'image
#Y_train : labels

#fonctions à créer :
#- fonction de normalisation des images : diviser tous les pixels par 255
#- fonction test : ne met pas à jour les poids sinon même systeme que fonction deroulement
#- fonction taux de réussite : à calculer à la fin des phases d'apprentissage et de test, doit être égal à 0.89 environ, reussite/(reussite+defaite)


#autres modifs :
#initialiser les poids à 0
#utiliser methode .ravel() pour modifier le contenu des listes dans x_train
#configurer le label : on choisit un chiffre d'entrainement, si pas le chiffre mettre 0 comme réponse sinon 1

from math import exp
import numpy as np

liste_images=[]

class perceptron:
    def __init__(self):
        self.biais=1
        self.label= 0 # importé avec l'image
        self.n=0.03 #taux d'apprentissage
        self.poids=list(np.random.uniform(0,1,28**2))
        self.observations=[]


    def deroulement(self, image):
        self.observations = image
        sum=self.attribution_poids()
        resultat=self.fonction_activation(sum)
        erreur=self.erreur(resultat)
        self.maj_poids(erreur)

    def fonction_activation(self, sum):
        if sum<0.5:
            return 0
        else:
            return 1

    def erreur(self,resultat):
        if self.label!=resultat :
            erreur=self.n(self.label-resultat)
            return erreur
        else :
            return 0

    def attribution_poids(self):
        sum=0
        for i in range(len(self.observations)):
            sum+= self.poids[i]*self.observations[i]
        return sum

    def maj_poids(self,erreur):
        for i in range(len(self.poids)):
            new_poids=self.poids[i]+self.n*erreur*self.observations[i]
            self.poids[i]=new_poids

Neurone=perceptron()
for image in liste_images:
    Neurone.deroulement(image)
#Neurone.deroulement()


#
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
    #

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

print(x_train[0])
print(y_train[0])
