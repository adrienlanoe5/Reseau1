from tkiteasy import *

class interface():
    def __init__(self):
        self.g=ouvrirFenetre(300,400)
        dico_objets={}


    def affichage(self):
        for j in range(3):
            for i in range(3):
                self.g.dessinerRectangle(i*80+25 ,100+70*j,50,40,"grey")
        self.g.attendreClic()


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