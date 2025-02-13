#X_train : liste à 2 niveaux
#1er niveau : une image
#2ème : plusieurs listes correspondant à chacune à un niveau de l'image
#Y_train : labels

#liste choses à faire:
#- dérivée fonction activation
#- cross entropy loss

import numpy as np

class reseau_neurones():
    def __init__(self, liste_neurones):
        self.nb_neurones =liste_neurones
        self.nb_couches=len(self.nb_neurones) #ensemble des couches cachées et la derniere
        self.liste_poids= self.initialisation_poids()
        self.reussite=0
        self.defaite=0

    def initialisation_poids(self):
        liste=[]
        mat_1=np.zeros(self.nb_neurones[0],28*28+1)
        liste.append(mat_1)
        for i in range(1,self.nb_couches):
            mat=np.zeros(self.nb_neurones[i],self.nb_neurones[i-1]+1)
            liste.append(mat)
        return liste

    def normalisation_image(self,image):
        for i in range(len(image)):
            image[i] = image[i] / 255
        return image

    def reset(self):
        self.reussite=0
        self.defaite=0

    def apprentissage(self,image,label_image):
        self.archi_resultats = []
        resultat_couche=self.normalisation_image(image)
        for i in range(self.nb_couches):
            resultat_couche=self.forward_propagation_produit_matriciel(i,resultat_couche)
            resultat_couche=self.fonction_activation(resultat_couche)
            self.archi_resultats.append(resultat_couche)
        resultat=self.softmax(resultat_couche)
        label_pred=str(np.argmax(resultat))
        erreur=self.erreur(resultat, label_pred, label_image)


    def forward_propagation_produit_matriciel(self, couche, inputs):
        vect_resultat=np.matmul(self.liste_poids[couche],inputs)
        new_vect=np.append(vect_resultat, [1])
        return new_vect

    def fonction_activation(self,x):
        return 1/(1 + np.exp(-x)) #sigmoide

    def softmax(self,liste):
        return np.exp(liste) / np.sum(np.exp(liste), axis=0)

    def erreur(self, resultat, label_pred, label_image): #a modifier
            if label_pred != label_image :
                self.defaite += 1
                return -resultat
            else:
                self.reussite += 1
                return 1-resultat


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


Neurone=reseau_neurones()
#phase apprentissage
for i in range (len(x_train)) :
    new_image=np.ravel(x_train[i])
    Neurone.apprentissage(new_image, y_train[i])

print(Neurone.taux_reussite())
Neurone.reset()

#phase de tests
#for i in range (len(x_test)) :
    #new_image=np.ravel(x_test[i])
    #Neurone.test(new_image, y_test[i])
#print(Neurone.reussite,Neurone.defaite)

#print(Neurone.taux_reussite())