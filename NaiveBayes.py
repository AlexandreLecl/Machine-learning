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
import math

class BayesNaif:

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		""" 
		self.probaClasse=[] # contient la proportion d'une classe 
		
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
		self.echantillon=train
		self.classeEchant=train_labels
		self.listClasses=[] 	# liste des différentes classes existentes
		lentrain=len(train)
		for classe in train_labels:
			if not classe in self.listClasses:
				self.listClasses.append(classe)
				self.probaClasse.append([0,classe])
			for k in range(0,len(self.probaClasse)):
				if classe == self.probaClasse[k][1]:
					self.probaClasse[k][0] +=1/lentrain

	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		lenEchant=len(self.echantillon)
		maxProba=0
		maxClasse=self.listClasses[0]
		for k in range(0,len(self.listClasses)):
			classe=self.listClasses[k]
			proba=self.probaClasse[k][0] #proba de la classe

			for i in range (0,len(x)):
				x_attribut=x[i]
				tempProba=0
				if type(x_attribut)==float:
					moyenne=0
					variance=0
					nbr=0
					for j in range (0,lenEchant): # calcul moyenne
						if self.classeEchant[j] == classe:
							moyenne += self.classeEchant[j][i]
							nbr +=1
					moyenne /= nbr
					for j in range (0,lenEchant): # calcul moyenne
						if self.classeEchant[j] == classe:
							variance += (self.classeEchant[j][i] - moyenne)**2
					variance /= nbr -1
					tempProba = 1/(math.sqrt(2*math.pi*variance))*math.exp(-((x_attribut-moyenne)**2)/(2*variance)) # proba loi normale
				else:
					for j in range (0,lenEchant):
						if self.classeEchant[j] == classe and self.echantillon[j][i]==x_attribut:
							tempProba+=1/lenEchant
				proba *= tempProba

			if proba > maxProba:
				maxProba=proba
				maxClasse=classe

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