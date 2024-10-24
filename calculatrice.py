from tkiteasy import *

class interface():
    def __init__(self):
        self.g=ouvrirFenetre(600,600)
        dico_objets={}


    def affichage(self):
        for i in range(10):
            self.g.dessinerDisque(100,500,30,"white")
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


