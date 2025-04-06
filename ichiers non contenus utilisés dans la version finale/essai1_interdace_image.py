import tkiteasy as tk
from essai1_reseau_neurones import activer_Neurone
from essai1_reseau_neurones import prediction_Neurone
import numpy as np



#interface _image
def interface_image(x_train, y_train,cote_entree=28):
    cote = cote_entree * 20
    hplus = 100
    g = tk.ouvrirFenetre(x=cote, y=cote + hplus)
    g.dessinerRectangle(x=0, y=0, l=cote, h=hplus, col='yellow')
    g.dessinerRectangle(x=0, y=hplus, l=cote, h=cote, col='white')
    txt = g.afficherTexte('cliquer pour dessiner', cote / 2, hplus / 2 + 1, col='black', sizefont=20)

    g.attendreClic() #attend que l'utilisateur clique
    g.changerTexte(txt,'dessin en cours')
    grande_matrice = [[0 for i in range(cote)] for j in range(cote)]  # matrice contenant l'ensemble des pixels dessinés par l'utilisateur : 1 si récupéré, 0 sinon
    while g.recupererClic()==None: #récupère les positions avant le clic de fin
        x, y = g.recupererPosition().x, g.recupererPosition().y
        grande_matrice[y-hplus][x]=1
        g.changerPixel(x, y, 'black')
        #colorie aussi le voisinnage
        g.changerPixel(x-1,y,'black') #gauche
        grande_matrice[y-hplus][x-1] = 1
        g.changerPixel(x+1, y, 'black') #droite
        grande_matrice[y-hplus][x-1] = 1
        g.changerPixel(x, y-1, 'black') #bas
        grande_matrice[y-hplus-1][x] = 1
        g.changerPixel(x, y+1, 'black') #haut
        grande_matrice[y-hplus+1][x] = 1

    matrice=[[0 for i in range(28)] for j in range(28)]
    for x in range(28):
        for y in range(28):
            # récupération de l'espace associé au pixel de la matrice
            lignes_matrice_pixel=grande_matrice[(cote//28)*x:(cote//28)*(x+1)]
            matrice_pixel=[]
            for i in range(cote//28):
                matrice_pixel.append(lignes_matrice_pixel[i][(cote // 28) * y: (cote // 28) * (y + 1)])
            #calcul de la coloration moyenne de la case
            a=np.sum(matrice_pixel)/(cote//28)**2
            #matrice[x][y]+=a*255 #annule la normalisation de l'image
    image=np.ravel(matrice)
    g.attendreClic()
    g.fermerFenetre()
    Neurone = activer_Neurone(x_train,y_train, nb_entrees=cote_entree**2,sans_entrainement=True)
    print(Neurone.taux_reussite())
    res=prediction_Neurone(Neurone,image)
    print(res)