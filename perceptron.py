from math import exp
import numpy as np

liste_images= []

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
        res=1 / (1 + exp(-sum))
        return res

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