#X_train : liste à 2 niveaux
#1er niveau : une image
#2ème : plusieurs listes correspondant à chacune à un niveau de l'image
#Y_train : labels

import numpy as np
from math import exp,tan
class perceptron:
    def __init__(self,n):
        self.biais=1
        self.n=n #taux d'apprentissage
        #self.poids=list(np.random.standard_normal(28**2+1))
        self.poids=[0 for i in range(28**2+1)]
        self.observations=[]
        self.label=3
        self.reussite=0
        self.defaite=0

    def reset(self):
        self.reussite=0
        self.defaite=0

    def normalisation_image_bruit(self,image):
        for i in range(len(image)):
            image[i] = image[i] / 255
            bruit=self.bruitage()
            image[i]=image[i]+bruit #bruitage
        image = np.append(image, [self.biais])
        return image

    def normalisation_image_bruit2(self,image):
        for i in range(len(image)):
            image[i] = image[i] / 255
            if 336<i<448:
                image[i] =0
        image = np.append(image, [self.biais])
        return image

    def normalisation_image(self,image):
        for i in range(len(image)):
            image[i] = image[i] / 255
        image = np.append(image, [self.biais])
        return image

    def apprentissage (self, image,label_image):
        self.observations = self.normalisation_image(image)
        sum=self.attribution_poids()
        resultat=self.fonction_activation(sum)
        erreur=self.erreur(resultat,label_image)
        self.maj_poids(erreur)

    def test(self, image,label_image):
        #self.observations = self.normalisation_image_bruit(image)
        #self.observations = self.normalisation_image_bruit2(image)
        self.observations = self.normalisation_image(image)
        sum=self.attribution_poids()
        resultat=self.fonction_activation(sum)
        self.erreur(resultat,label_image)

    def bruitage (self):
        #bruitage gaussien
        ecart_type=0.4
        bruit=np.random.normal(0,ecart_type,1) +np.random.uniform(-ecart_type,ecart_type,1)
        #bruitage loi uniforme
        #bruit=np.random.uniform(-ecart_type,ecart_type,1)
        return bruit

    def fonction_activation(self, sum):
        #nb = 1 / (1 + exp(-sum))  # fonction sigmoïde
        #nb=tan(sum) #fonction tangente
        #nb=(exp(sum)-exp(-sum))/(exp(sum)+exp(-sum)) #fonction tangente hyperbolique

        if sum<0:
        #if nb<0.5:
            return 0
        else:
           return 1


    def erreur(self,resultat,label_image):
        #print("self.label",self.label,'label_image',label_image)
        #print('resultat',resultat)
        if self.label!=label_image and resultat==1:
            self.defaite+=1
            #print('defaite',self.defaite,'reussite',self.reussite)
            return -1
        elif self.label == label_image and resultat == 0:
            self.defaite+=1
            #print('defaite', self.defaite, 'reussite', self.reussite)
            return 1
        else :
            self.reussite+=1
            #print('defaite', self.defaite, 'reussite', self.reussite)
            return 0

    def attribution_poids(self):
        sum=0
        for i in range(len(self.observations)):
            sum+= self.poids[i]*self.observations[i]
        return sum/(len(self.observations)+1)

    def maj_poids(self,erreur):
        for i in range(len(self.poids)):
            new_poids=self.poids[i]+self.n*erreur*self.observations[i]
            #new_poids = self.poids[i] + self.n * (erreur**2) * self.observations[i] #distance au carré
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




for j in range(1,10):
    Neurone = perceptron(j*10**-2)

    # phase apprentissage
    for i in range(len(x_train)):
        new_image = np.ravel(x_train[i])
        Neurone.apprentissage(new_image, y_train[i])

    print('reussite',Neurone.taux_reussite(),'n',j*(10**-2),'apprentissage')
    Neurone.reset()

    # phase de tests
    for i in range(len(x_test)):
        new_image = np.ravel(x_test[i])
        Neurone.test(new_image, y_test[i])

    print('reussite',Neurone.taux_reussite(),'n',j*(10**-2),'test')
    # print(Neurone.poids)

# n optimal 0.05