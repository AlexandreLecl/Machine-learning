import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB  
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres

nbrVoisins=11    # Meilleur valeur selon la validation croisée : Iris: 11 ; Wine: 1 ; Abalone: 9
train_ratio=.75
classifieur = 1  # 0= Knn ; autres = NaiveBayes
dataset = 2     # 0 = iris ; 1 = wine ; autres = abalone

# Initialisez/instanciez vos classifieurs avec leurs paramètres
if classifieur==0:
    classifieurKnn=Knn.Knn(nbrVoisins)
    scikitKNN=KNeighborsClassifier(n_neighbors=nbrVoisins)
else:
    classifieurNaifBayes=NaiveBayes.BayesNaif()
    scikitBayes=GaussianNB()


# Charger/lire les datasets
if dataset ==0:
    train,train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
    print("Iris dataset:\n")
elif dataset ==1 :
    train,train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
    print("Wine dataset:\n")
else :
    train,train_labels, test, test_labels = load_datasets.load_abalone_dataset(train_ratio)
    print("Abalone dataset:\n")

# Entrainez votre classifieur
if classifieur==0:
    classifieurKnn.train(train,train_labels)
    scikitKNN.fit(train,train_labels)
else:
    classifieurNaifBayes.train(train,train_labels)
    scikitBayes.fit(train,train_labels)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
if classifieur==0:
    confusion,accuracy,precison,rappel,fscore=classifieurKnn.evaluate(train,train_labels)
    scikitAccuracy=scikitKNN.score(train,train_labels)
else:
    confusion,accuracy,precison,rappel,fscore=classifieurNaifBayes.evaluate(train,train_labels)
    scikitAccuracy=scikitBayes.score(train,train_labels)

print("Performance en train :\n")
print("Matrice de confusion :\n",confusion,"\n")
print("Exactitude:",accuracy,"\n")
print("Précision:",precison,"\n")
print("Rappel:",rappel,"\n")
print("F1-score:",fscore,"\n")
print("scikit exactitude:",scikitAccuracy,"\n")

# Tester votre classifieur
if classifieur==0:
    confusion,accuracy,precison,rappel,fscore=classifieurKnn.evaluate(test,test_labels)
    scikitAccuracy=scikitKNN.score(test,test_labels)
else:
    confusion,accuracy,precison,rappel,fscore=classifieurNaifBayes.evaluate(test,test_labels)
    scikitAccuracy=scikitBayes.score(test,test_labels)

"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
print("Performance en test :\n")
print("Matrice de confusion :\n",confusion,"\n")
print("Exactitude:",accuracy,"\n")
print("Précision:",precison,"\n")
print("Rappel:",rappel,"\n")
print("F1-score:",fscore,"\n")
print("scikit exactitude:",scikitAccuracy,"\n")




