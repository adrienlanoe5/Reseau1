from math import exp

class perceptron:
    def __init__(self):
        self.biais=1
        self.label= 0 # importé avec l'image
        self.n=0.03 #taux d'apprentissage
    def deroulement(self):

    def fonction_activation(self, sum):
        res=1 / (1 + math.exp(-sum))
        return res

    def erreur(self,resultat):
        if self.label!=resultat
            erreur=self.n(self.label-resultat)
            return erreur
        else :
            return 0

    def poids(self):

    def maj_poids(self):





Neurone=perceptron()
Neurone.deroulement()