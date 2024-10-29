from tkiteasy import *

liste_symboles_dopérations=["*","/","+","-"]
class interface():
    def __init__(self):
        self.g=ouvrirFenetre(300,400)
        dico_objets={}


    def affichage(self):
        for j in range(3):
            for i in range(3):
                self.g.dessinerRectangle(i*80+25 ,100+70*j,50,40,"grey")
                self.g.afficherTexte(str((i+1)+3*j), i*80+50 ,120+70*j, "white", 25)
        self.g.dessinerRectangle(1 * 80 + 25, 100 + 70 * 3, 50, 40, "grey")
        self.g.afficherTexte("0", 1 * 80 + 50, 120 + 70 * 3, "white", 25)
        for i in range (4):
            self.g.dessinerCercle(i * 40 + 25, 80, 30, "grey")
            self.g.afficherTexte(liste_symboles_dopérations[i], i * 80 + 50, 120 + 70 * j, "white", 25)



    def addition(self,a,b):
        return a+b

    def division(self,a,b):
        return a/b

    def multiplication(self,a,b):
        return a*b

    def soustraction(self,a,b):
        return a-b

    def garderminchiffre(self,nombre): # pour avoir un nombre fini de chiffres à afficher sur la calculatrice
        pass

I=interface()
I.affichage()