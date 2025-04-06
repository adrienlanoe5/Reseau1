# imporation des données
from essai1_bdd_lettres import importer_bdd_lettres
from essai1_bdd_lettres import recherche_label
from  essai1_reseau_neurones import activer_Neurone
from essai1_reseau_neurones import prediction_Neurone
from essai1_bdd_lettres import conversion_prevision_caractere
from essai1_reseau_neurones import boucle


## Etape 1 : initialsation des données d'entrainement et de test
(lettres_train, lettres_label_train), (lettres_test, lettres_label_test)=importer_bdd_lettres()

## Etape 2 et 3 : entrainement du réseau et phase de test. On a enlevé 33 à chaque élément de y_train et y_test

Neurone=activer_Neurone(lettres_train,lettres_label_train,lettres_test,lettres_label_test,nb_entrees=32*32,liste_nb_neurones=[5, 8, 6, 93])

## Etape 4 : récupération du mot de l'utilisateur
mot=input("Tapez le mot à prédire ")
liste_lettres=[]
for i in range(len(mot)):
    liste_lettres.append(mot[i])

## Etape 5 : calcul de la prédiction du réseau de neurones
liste_label=recherche_label(liste_lettres,lettres_label_train)
liste_prediction=prediction_Neurone([lettres_train[label] for label in liste_label],Neurone)
liste_lettres_predites=conversion_prevision_caractere(liste_prediction)
mot_predit=''
for elem in liste_lettres_predites:
    mot_predit+=elem
print('les caractères prédits sont :',mot_predit)

## Etape 6 : Mise en valeur des plans de test

taux_apprentissage = [0.01, 0.03, 0.06, 0.25, 0.5, 0.75, 1]
type_fonction_acti = ["sigmoide", "tangente hyperbolique", "tangente", "selu"]
liste_neurones = [[32*32,68,34,5],[32*32,9,20,30]]

boucle(taux_apprentissage,"taux_apprentissage",lettres_train, lettres_label_train, lettres_test, lettres_label_test,nb_entrees=32*32)
boucle(type_fonction_acti,"fonction_activation",lettres_train, lettres_label_train, lettres_test, lettres_label_test,nb_entrees=32*32)
boucle(liste_neurones,"neurones",lettres_train, lettres_label_train, lettres_test, lettres_label_test,nb_entrees=32*32)