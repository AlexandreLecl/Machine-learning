"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
from math import sqrt
import load_datasets

class Knn: 

	def __init__(self,kVoisins, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.weigths=[]
		self.kVoisins=kVoisins
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		# établir liste échantillon + liste poids (pondéré par 1/diff min max au carré)
		self.echantillon=train
		self.classeEchant=train_labels
		self.listClasses=[] 	# liste des différentes classes existentes
		for classe in train_labels:
			if not classe in self.listClasses:
				self.listClasses.append(classe)
		

		#Calcul des poids :
		minMax=[]
		first=train[0]
		for attribut in first:
			minMax.append((attribut,attribut))

		for x in train:
			i=0
			for attribut in x:
				if type(attribut)==float:
					minAtt,maxAtt=minMax[i]
					if attribut < minAtt:
						minMax[i]=(attribut,maxAtt)
					elif attribut > maxAtt:
						minMax[i]=(minAtt,attribut)
				i+=1
		
		for paire in minMax:
			minAtt,maxAtt=paire
			if type(minAtt)==float:
				x=(minAtt-maxAtt)**2
				if x ==0:
					self.weigths.append(0)
				else:
					self.weigths.append(1/x)
			else:
				self.weigths.append(1)
    
	def dist(self,x,y):
		"""
		Calcul la distance entre deux éléments en fonction de leurs attributs
		"""
		distance=0.0
		for i in range(len(x)):
			if type(x[i])==float:
				distance += self.weigths[i] * (x[i]-y[i])**2
			elif x[i]!=y[i]:
				distance += self.weigths[i]
		distance=sqrt(distance)
		return distance

	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		#Détermination des plus proches voisins
		listKPlusProche=[]
		for i in range(0,len(self.echantillon)):
			y=self.echantillon[i]
			classe=self.classeEchant[i]
			distance=self.dist(x,y)
			if len(listKPlusProche)<self.kVoisins:
				listKPlusProche.append((distance,classe))
			else:
				indexMax=0
				maxDist=listKPlusProche[0][0]
				for j in range(0,len(listKPlusProche)):
					tempDist=listKPlusProche[j][0]
					if maxDist< tempDist:
						indexMax=j
						maxDist=tempDist
				if distance < maxDist:
					listKPlusProche[indexMax]=(distance,classe)
		
		#Décision de la classe
		countClasse=[]
		for classe in self.listClasses:
			countClasse.append([classe,0])
		for voisins in listKPlusProche:
			classe=voisins[1]
			for x in countClasse:
				if x[0]==classe:
					x[1] += 1

		maxClasse=countClasse[0][0]
		maxCompt=countClasse[0][1]
		for i in range(0,len(countClasse)):
			tempCompt=countClasse[i][1]
			if tempCompt > maxCompt :
				maxCompt=tempCompt
				maxClasse=countClasse[i][0]
		
		return maxClasse

	def calcPerf(self,confusion,n):
		"""
		c'est la méthode qui calcule les mesures de performances à partir de la matrice de confusion
		confusion : Matrice de confusion de taille nbrClasses x nbrClasses
		n : nombre d'exemples du dataset à partir duquel la matrice de confusion est calculée
		"""
		nbrClasses=len(self.listClasses)
		totExactitude=0
		totPrecision=0
		totRappel=0

		for ligne in range(0,nbrClasses): #classe réelle
			sommeligne=0
			sommecolonne=0
			rappel=0
			for colonne in range(0,nbrClasses): #classe prédite
				sommeligne += confusion[ligne,colonne] # Vrai Positif ou Faux Négatif
				if ligne == colonne:
					rappel = confusion[ligne,colonne] # Vrai Positif
					totExactitude += confusion[ligne,colonne]
			rappel /= sommeligne
			totRappel += rappel
		totExactitude /= n
		totRappel /= nbrClasses

		for colonne in range(0,nbrClasses):	#classe prédite
			sommecolonne=0
			précision=0
			for ligne in range(0,nbrClasses): #classe réelle
				sommecolonne += confusion[ligne,colonne] #Faux positif et Vrai positif
				if ligne == colonne:
					précision = confusion[ligne,colonne] # Vrai Positif
			précision /= sommecolonne
			totPrecision += précision
		
		totPrecision /= nbrClasses
		totFScore= totPrecision * totRappel /(totPrecision + totRappel)

		return totExactitude,totPrecision,totRappel,totFScore
	
	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
        # faire prédiction pour chaque données et voir rslt
		n,m=X.shape
		nbrClasses=len(self.listClasses)
		confusion=np.zeros((nbrClasses,nbrClasses))

		for i in range(0,n):
			classePredict=self.predict(X[i])
			for j in range (0,nbrClasses):
				if self.listClasses[j]==classePredict:
					ligne=j
				if self.listClasses[j]==y[i]:
					colonne=j
			confusion[ligne,colonne] +=1
		
		exactitude,precision,rappel,fscore=self.calcPerf(confusion,n)

		return confusion,exactitude,precision,rappel,fscore

	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.
	