import tkiteasy as tk
import numpy as np

cote=28*20
g=tk.ouvrirFenetre(x=cote,y=cote)
g.dessinerRectangle(x=0, y=0, l=cote, h=cote, col='white')
liste=[] #ensemble des positions dessinées par l'utilisateur
while g.recupererClic()==None: #attend que l'utilisateur clique
    pass
while g.recupererClic()!=None: #attend la fin du clic
    pass
while g.recupererClic()==None: #récupère les positions avant le clic de fin
    liste.append((g.recupererPosition().x,g.recupererPosition().y)) #récupére l'ensemble des positions de l'utilisateur penddant le clic
grande_matrice=[[0 for i in range(cote)] for j in range(cote)] #matrice contenant l'ensemble des pixels dessinés par l'utilisateur : 1 si récupéré, 0 sinon
for elem in liste:
    x,y=elem
    grande_matrice[y][x]=1
    g.changerPixel(x, y, 'black')

print(grande_matrice)

matrice=[[0 for i in range(28)] for j in range(28)]
for x in range(28):
    for y in range(28):
        # récupération de l'espace associé au pixel de la matrice
        a=(cote//28)*x
        b=(cote//28)*(x+1)
        c=(cote//28)*y
        d=cote//28*(y+1)
        lignes_matrice_pixel=grande_matrice[(cote//28)*x:(cote//28)*(x+1)]
        matrice_pixel=[]
        for i in range(cote//28):
            matrice_pixel.append(lignes_matrice_pixel[i][(cote // 28) * y: (cote // 28) * (y + 1)])
        #calcul de la coloration moyenne de la case
        print(np.sum(matrice_pixel))
        a=np.sum(matrice_pixel)/(cote//28)**2
        matrice[x][y]+=a





g.attendreClic()

