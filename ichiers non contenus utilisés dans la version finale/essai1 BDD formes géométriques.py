import numpy as np
import os
from PIL import Image

def ChargementBase(dossier):
    images = []
    labels = []
    noms_images = []

    # Parcours de tous les fichiers du dossier
    for fichier in os.listdir(dossier):
        chemin_complet = os.path.join(dossier, fichier)

        # Vérifier si c'est bien une image
        if fichier.endswith((".jpg", ".png", ".jpeg")):
            # Chargement et transformation de l'image
            image = Image.open(chemin_complet)
            image = image.resize((28, 28))
            image = image.convert("L")  # Convertir en niveaux de gris
            image_array = np.asarray(image).flatten()  # Aplatir en 1D

            # Définition du label basé sur le nom de l’image
            if "circle" in fichier.lower():
                label = 1
            elif "square" in fichier.lower():
                label = 2
            elif "triangle" in fichier.lower():
                label = 3
            elif "kite" in fichier.lower():
                label = 4
            elif "parallelogram" in fichier.lower():
                label = 5
            elif "rectangle" in fichier.lower():
                label = 6
            elif "rhombus" in fichier.lower():
                label = 7
            elif "trapezoid" in fichier.lower():
                label = 8
            else:
                label = 0  # Label par défaut si inconnu

            # Stocker l'image, son label et son nom
            images.append(image_array)
            labels.append(label)
            noms_images.append(fichier)

    # Conversion en tableaux NumPy
    images = np.array(images, dtype=np.float32)  # Optionnel : Normalisation possible
    labels = np.array(labels, dtype=np.int32)

    images_train = images[:3000]
    images_test = images[3000:]
    labels_train = labels[:3000]
    labels_tests = labels[3000:]

    print(f"Base chargée avec {len(images)} images")
    print(f"Noms des premières images : {noms_images[:5]}")
    print(f"Labels des premières images : {labels[:5]}")

    return (images_train,labels_train),(images_test,labels_tests)

#images_filepath="donnees_entrainement_formes/Data_forme"
#(images_train,labels_train),(images_test,labels_tests)= ChargementBase(images_filepath)
