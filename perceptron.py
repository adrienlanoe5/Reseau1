class perceptron:
    def __init__(self):
        self.w=[1,1,0.5,0.8,0.7,0.3,0.09,0.099]


    def deroulement(self):


    def fonction_activation(self):

    def erreur(self):

    def poids(self):

    def maj_poids(self):


    def calcul_valeur(self,entree): # l'etrée est une liste de 0 et de 1
        sum=0
        for i in range(len(entrée)):
            sum+=self.w[i]*entree[i]
        return self.fonction_activation (sum)






Neurone=perceptron()
Neurone.deroulement()