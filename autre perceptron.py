#X_train : liste à 2 niveaux
#1er niveau : une image
#2ème : plusieurs listes correspondant à chacune à un niveau de l'image
#Y_train : labels


#- fonction taux de réussite : à calculer à la fin des phases d'apprentissage et de test, doit être égal à 0.89 environ,
#autres modifs :

#
# MNIST Dataset

import numpy as np
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



import numpy as np

liste_images=[]

class perceptron:
    def __init__(self):
        self.biais=1
        self.n=0.03 #taux d'apprentissage
        self.poids=[0 for i in range(28**2)]
        self.observations=[]
        self.label=3
        self.attendu=0

    def normalisation_image(self,image):
        for i in range (len(image)):
            image[i]=image[i]/255
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
        return self.attendu==resultat

    def fonction_activation(self, sum):
        if sum<0:
            return 0
        else:
            return 1

    def erreur(self,resultat,label_image):
        if label_image==self.label:
            self.attendu=1
        else:
            self.attendu=0
        return self.attendu-resultat

    def attribution_poids(self):
        sum=0
        for i in range(len(self.observations)):
            sum+= self.poids[i]*self.observations[i]
        sum+=self.biais
        return sum

    def maj_poids(self,erreur):
        for i in range(len(self.poids)):
            new_poids=self.poids[i]+self.n*erreur*self.observations[i]
            self.poids[i]=new_poids




Neurone=perceptron()

#phase apprentissage
for i in range (len(x_train)) :
    new_image=np.ravel(x_train[i])
    Neurone.apprentissage(new_image, y_train[i])

#phase de tests for i in range (len(x_train)) :
n_reussites=0
for i in range (len(x_test)) :
    new_image=np.ravel(x_test[i])
    if Neurone.test(new_image, y_test[i]):
        n_reussites+=1
print("pourcentage d'erreur",(1-n_reussites/len(x_test))*100)

#phase test
#for image in x_test:
#    new_image=image.np.ravel()
#    Neurone.test(new_image, y_test[image])
#Neurone.taux_reussite()