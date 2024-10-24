from tkiteasy import *

class interface():
    def __init__(self):
        self.g=ouvrirFenetre(600,600)

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

interface()
ouvrirFenetre(100,100)