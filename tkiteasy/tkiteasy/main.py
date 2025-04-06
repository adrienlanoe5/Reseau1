#coding: utf-8

###### ouverture de fenêtre
g = ouvrirFenetre(800,600)

txthaut = g.afficherTexte("Cliquez pour débuter la démonstration",400,50,"white")
g.actualiser()
g.attendreClic()

###### fonctions géométriques standard
# dessinerDisque()
g.changerTexte(txthaut,"Fonction dessinerDisque(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
d1 = g.dessinerDisque(600,500,30,"grey")

# dessinerCercle()
g.changerTexte(txthaut,"Fonction dessinerCercle(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
g.dessinerCercle(100,500,30,"purple")

# dessinerLigne()()
g.changerTexte(txthaut,"Fonction dessinerLigne(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
g.dessinerLigne(200,20,750,340,"pink")

# dessinerRectangle()
g.changerTexte(txthaut,"Fonction dessinerRectangle(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
r1 = g.dessinerRectangle(699,499,100,50,"yellow")
r2 = g.dessinerRectangle(442,416,40,80,"green")

# supprimer()
g.changerTexte(txthaut,"Fonction supprimer(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
g.supprimer(r2)

# afficherTexte()
g.changerTexte(txthaut,"Fonction afficherTexte(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
txt = g.afficherTexte("coucou",400,300,"red", 45)

# afficherTexte()
g.changerTexte(txthaut,"Fonction changerTexte() & changerCouleur(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
g.changerTexte(txt,"ouep")
g.changerCouleur(txt,"blue")



# afficherImage()()
g.changerTexte(txthaut,"Fonction afficherImage(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
pacman = g.afficherImage(400,300,"./pacman.png")

# deplaceer()
g.changerTexte(txthaut,"Fonction deplacer(): cliquez pour continuer")
g.actualiser()
g.attendreClic()
for i in range(100):
   g.deplacer(pacman,0,-1)
   g.actualiser()
   g.pause()

# attendreClic et coordonnées
txtbas = g.afficherTexte(f"coordonnées pacman {pacman.x}, {pacman.y}",400,550,"white")
g.changerTexte(txthaut,"Fonction bloquante attendreClic(): cliquez pour continuer")
g.actualiser()
clic = g.attendreClic()
g.changerTexte(txtbas,f"Vous avez cliqué en ({str(clic.x)},{str(clic.y)})")
g.actualiser()
g.attendreClic()

# interaction bloquante souris
g.changerTexte(txthaut,"Fonction attendreClic()")
g.changerTexte(txtbas,"Cliquez pour déplacer pacman, bouton droit pour quitter")
while(True):
    clic = g.attendreClic()
    if clic.num==3: # 1 = bouton gauche, 3 = bouton droit
        break
    g.deplacer(pacman,clic.x-pacman.x, clic.y-pacman.y) # déplacement relatif!
    g.actualiser()

# interaction bloquante clavier
g.changerTexte(txthaut,"Fonctions bloquante attendreTouche()")
g.changerTexte(txtbas,"Jouer avec les flèches du clavier, 'q' pour quitter")
while(True):
    touche = g.attendreTouche()
    if touche=="q":
        break
    elif touche=="Right" and pacman.x<800:
        g.deplacer(pacman,5,0)
    elif touche=="Left" and pacman.x>=5:
        g.deplacer(pacman,-5,0)
    elif touche=="Up" and pacman.y>=5:
        g.deplacer(pacman,0,-5)
    elif touche=="Down"and pacman.y<600:
        g.deplacer(pacman,0,5)

    g.actualiser()

# interaction non bloquante souris
g.changerTexte(txthaut,"Fonction non bloquante recupererClic() et recupererPosition(), clic pour quitter")
while(True):
    clic = g.recupererClic()
    pos = g.recupererPosition()
    if clic!=None and clic.num==1:
        break
    if pos!=None:
        g.deplacer(pacman,pos.x-pacman.x, pos.y-pacman.y) # déplacement relatif!
    g.changerTexte(txtbas,f"Pacman en ({str(pacman.x)},{str(pacman.y)})")
    g.actualiser()


# fermerFenetre()
g.changerTexte(txthaut,"Fin de la démonstration: cliquez pour terminer")
g.attendreClic()
g.fermerFenetre()
