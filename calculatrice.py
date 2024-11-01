from tkiteasy import *

operations = ["*", "/", "+", "-"]


class interface():
    def __init__(self):
        self.g = ouvrirFenetre(350, 400)
        self.memoire = []  # utilisee pour la phase de calcul
        self.memoire_affichage = []  # utilisee pour l'affichage dans la fenetre
        self.objet = {}  # stocker les boutons chiffres et operations

    def affichage(self):
        # chiffres
        for j in range(3):
            for i in range(3):
                bouton = self.g.dessinerRectangle(i * 80 + 25, 100 + 70 * j, 50, 40, "grey")
                chiffre = self.g.afficherTexte(str((i + 1) + 3 * j), i * 80 + 50, 120 + 70 * j, "white", 25)
                self.objet[bouton] = (i + 1) + 3 * j
                self.objet[chiffre] = (i + 1) + 3 * j

        # bouton 0
        bouton = self.g.dessinerRectangle(1 * 80 + 25, 100 + 70 * 3, 50, 40, "grey")
        chiffre = self.g.afficherTexte("0", 1 * 80 + 50, 120 + 70 * 3, "white", 25)
        self.objet[bouton], self.objet[chiffre] = 0, 0

        # bouton entrer
        self.entrer_carre = self.g.dessinerRectangle(2 * 80 + 25, 100 + 70 * 3, 50, 40, "grey")
        self.entrer_lettres = self.g.afficherTexte("Entrer", 2 * 80 + 50, 120 + 70 * 3, "white", 10)

        # bouton quitter
        self.quitter_carre = self.g.dessinerRectangle(25, 100 + 70 * 3, 60, 40, "grey")
        self.quitter_lettres = self.g.afficherTexte("Quitter", 55, 120 + 70 * 3, "white", 10)

        # bouton effacer
        self.effacer_carre = self.g.dessinerRectangle(3 * 80 + 25, 100 + 70 * 3, 60, 40, "grey")
        self.effacer_lettres = self.g.afficherTexte("Effacer", 3 * 80 + 55, 120 + 70 * 3, "white", 10)

        # operations
        for i in range(4):
            bouton = self.g.dessinerDisque(i * 55 + 50, 75, 20, "grey")
            op = self.g.afficherTexte(operations[i], i * 55 + 50, 75, "white", 25)
            self.objet[bouton] = operations[i]
            self.objet[op] = operations[i]

    def deroulement(self):  # appel des fonctions et gestion des fonctionnalites
        self.affichage()
        stop = False
        while stop == False:
            clic = self.g.attendreClic()
            x = self.g.recupererObjet(clic.x, clic.y)

            if x == self.quitter_lettres or x == self.quitter_carre:  # quitter le programme
                self.fin()

            elif x == self.entrer_carre or x == self.entrer_lettres:  # declenchement phase de calcul
                self.resultat()

            elif x == self.effacer_carre or x == self.effacer_lettres:  # effacer tout
                self.superclean()

            elif x in self.objet:
                for objet in self.memoire_affichage:
                    self.g.deplacer(objet, -15, 0)
                self.memoire.append(self.objet[x])
                obj_affiche = self.g.afficherTexte(str(self.objet[x]), 300, 30, col='white', sizefont=25)
                self.memoire_affichage.append(obj_affiche)

    def superclean(self):  # effacer tout et reinitialisation
        self.g.supprimerTout()
        self.memoire = []
        self.memoire_affichage = []
        self.affichage()

    def fin(self):  # fermeture programme
        self.g.fermerFenetre()

    def resultat(self):
        pass

    def addition(self, a, b):
        return a + b

    def division(self, a, b):
        return a / b

    def multiplication(self, a, b):
        return a * b

    def soustraction(self, a, b):
        return a - b

    def garderminchiffre(self, nombre):  # pour avoir un nombre fini de chiffres à afficher sur la calculatrice
        pass


I = interface()
I.deroulement()



