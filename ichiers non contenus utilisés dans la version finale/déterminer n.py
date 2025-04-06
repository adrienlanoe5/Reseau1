#X_train : liste à 2 niveaux
#1er niveau : une image
#2ème : plusieurs listes correspondant à chacune à un niveau de l'image
#Y_train : labels
#fonction taux de réussite doit être égal à 0.89 environ

#Choses à faire pendant les vacances:
#faire des tests meilleurs paramètres/choix (chiffres...)
#faire des bruits dans les donnees de tests
#- entre 1 et 4 pixels à modifier
#- paramètre gaussien qui influence l'ensemble des pixels
#centrer le paramètre selon une N(0,1) puis N(1,1) et autre
import numpy as np

class perceptron:
    def __init__(self,n):
        self.biais=1
        self.n=n #taux d'apprentissage
        #self.poids=list(np.random.uniform(0,1,28**2+1))
        self.poids=[0 for i in range(28**2+1)]
        self.observations=[]
        self.label=3
        self.reussite=0
        self.defaite=0

    def reset(self):
        self.reussite=0
        self.defaite=0

    def normalisation_image(self,image):
        for i in range(len(image)):
            image[i]=image[i]/255
        image=np.append(image,[self.biais])
        return image

    def apprentissage (self, image,label_image):
        self.observations = self.normalisation_image(image)
        sum=self.attribution_poids()
        resultat=self.fonction_activation(sum)
        erreur=self.erreur(resultat,label_image)
        self.maj_poids(erreur)

    def test(self, image,label_image):
        self.observations = self.normalisation_image(image)
        sum=self.attribution_poids()
        resultat=self.fonction_activation(sum)
        self.erreur(resultat,label_image)

    def fonction_activation(self, sum):
        if sum<0.5:
            return 0
        else:
            return 1

    def erreur(self,resultat,label_image):
        if self.label!=label_image and resultat==1:
            self.defaite+=1
            return -1
        elif self.label == label_image and resultat == 0:
            self.defaite+=1
            return 1
        else :
            self.reussite+=1
            return 0

    def attribution_poids(self):
        sum=0
        for i in range(len(self.observations)):
            sum+= self.poids[i]*self.observations[i]
        return sum/(len(self.observations)+1)

    def maj_poids(self,erreur):
        for i in range(len(self.poids)):
            new_poids=self.poids[i]+self.n*erreur*self.observations[i]
            self.poids[i]=new_poids

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

training_images_filepath = '../Reseaudeneurones/archive/t10k-images.idx3-ubyte'
training_labels_filepath = '../Reseaudeneurones/archive/t10k-labels.idx1-ubyte'
test_images_filepath = '../Reseaudeneurones/archive/train-images.idx3-ubyte'
test_labels_filepath = '../Reseaudeneurones/archive/train-labels.idx1-ubyte'


# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

n_liste=[0.27+0.01*x for x in range(7)]
liste_resultats=[]

for elem in n_liste:
    Neurone=perceptron(elem)

    #phase apprentissage
    for i in range (len(x_train)) :
        new_image=np.ravel(x_train[i])
        Neurone.apprentissage(new_image, y_train[i])

    Neurone.reset()

    #phase de tests
    for i in range (len(x_test)) :
        new_image=np.ravel(x_test[i])
        Neurone.test(new_image, y_test[i])

    print(Neurone.taux_reussite())
    liste_resultats.append(Neurone.taux_reussite())

print(liste_resultats)

max=0
nmax=0
for i in range (len(liste_resultats)):
    if liste_resultats[i]>max:
        nmax=n_liste[i]
        max=liste_resultats[i]

print(nmax)

#n trouvé 0.29 avec  0.8991833333333333 de taux de réussite