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


    def calcul_valeur(self,entree): # l'etrée est une liste de 0 et de 1
        sum=0
        for i in range(len(entrée)):
            sum+=self.w[i]*entree[i]
        return self.fonction_activation (sum)

    def entrainement(self,entree,resultatattendu):
        erreur= resultatattendu - calcul_valeur(entree)
        for i in range(len(w)):





Neurone=perceptron()
Neurone.deroulement()