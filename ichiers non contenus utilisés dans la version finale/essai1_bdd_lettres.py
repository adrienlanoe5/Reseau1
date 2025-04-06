import os

import cv2 #installer le package Opencv
import numpy as np

from os.path import join

class bdd(): #Utiliser la fonction sortie
    def __init__(self):
        return None
    def verify_archive_structure(self,):
        """Vérifie la structure de l'archive extraite"""
        print("\nVérification de la structure des fichiers...")
        extract_dir = "../DataSetCreator/handwritting/curated_data"
        found_files = []

        # Recherche récursive de fichiers images
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    found_files.append(join(root, file))

        if not found_files:
            print("Aucune image trouvée. Structure actuelle :")
            os.system(f"tree -L 3 {extract_dir} || ls -R {extract_dir}")
        else:
            print(f"Found {len(found_files)} image files")

        return found_files


    def load_data_alternative(self):
        """Charge les données avec une structure inconnue"""
        X, y = [], []
        extract_dir = "split/curated_data"
        found_files = self.verify_archive_structure()

        for img_path in found_files:
            try:
                # Essai d'extraction du label depuis le chemin
                dir_name = os.path.basename(os.path.dirname(img_path))
                if dir_name.isdigit():
                    char_code = int(dir_name)
                else:
                    # Si les dossiers ne sont pas des codes ASCII
                    char_code = ord(os.path.basename(img_path)[0])  # Premier caractère du nom de fichier

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(cv2.resize(img, (32, 32)))
                    y.append(char_code)
            except Exception as e:
                print(f"Erreur sur {img_path}: {e}")

        self.X=np.array(X)
        self.Y=np.array(y)

        return self.X, self.Y


    #renvoie les datasets sous formes: x_train,y_train,x_test,y_test
    #Pour utiliser dans le résseau, il est important d'avoir les indices par raport à 0: faire -33 sur les labels
    def sortie(self):
        self.load_data_alternative()
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        x_shuffled = self.X[indices]
        y_shuffled = self.Y[indices]

        split_idx = int(0.8 * len(self.X))
        x_train=x_shuffled[:split_idx]
        y_train=y_shuffled[:split_idx]
        x_test=x_shuffled[split_idx:]
        y_test=y_shuffled[split_idx:]
        # enlève 33 à tous les éléments de y_train et à tous les éléments de y_test pour faire en sorte que tous les labels soient compris entre 0 et 93
        y_train-=33
        y_test-=33
        return (x_train, y_train), (x_test, y_test)

def importer_bdd_lettres():
    ma_bdd = bdd()
    return ma_bdd.sortie()

def recherche_label(liste_caracteres,y_train):

    # convertit les caractères en leur numéro ascii-33
    liste_numeros_caracteres=[]
    asciiDict = {chr(i): i - 33 for i in range((128))}
    for caract in liste_caracteres:
        liste_numeros_caracteres.append(asciiDict[caract])

    #recherche les labels et les stocke dans liste_labels
    liste_labels = [None for i in range(len(liste_caracteres))]
    j=0
    while None in liste_labels: #tant qu'il existe au moins un label qui n'a pas été récupéré
        if y_train[j] in liste_numeros_caracteres:
            for i in range(len(liste_numeros_caracteres)):
                if liste_numeros_caracteres[i]==y_train[j]:
                    liste_labels[i]=j
        j+=1
    return liste_labels

def conversion_prevision_caractere(liste_prediction):
    liste_caracteres=[]
    for pred in liste_prediction:
        liste_caracteres.append(chr(pred+33))
    return liste_caracteres