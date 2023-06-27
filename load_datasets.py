import numpy as np
import random

def getDataFromLine(line):
    listData=[]
    buffer=""
    for character in line:
        if character==',' or character=='\n':
            listData.append(buffer)
            buffer=""
        else:
            buffer += character
        lastCharacter=character
    if lastCharacter != '\n':
        #cas de la dernière ligne du fichier 
        listData.append(buffer)
    return (listData,len(listData))

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')
    
    
    # TODO : le code ici pour lire le dataset
    
    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
    lendata=sum(1 for line in f)
    f.close()
    f = open('datasets/bezdekIris.data', 'r')
    data=np.zeros((lendata,5))
    i=0
    for line in f:
        listData,lenList=getDataFromLine(line)
        
        longSep=float(listData[0])
        largSep=float(listData[1])
        longPet=float(listData[2])
        largPet=float(listData[3])
        label=conversion_labels[listData[4]]

        data[i,0]=longSep
        data[i,1]=largSep
        data[i,2]=longPet
        data[i,3]=largPet
        data[i,4]=label
        i +=1
    f.close()
    np.random.shuffle(data)
    lentrain=int(train_ratio*lendata)
    train=data[0:lentrain,0:4]
    train_labels=data[0:lentrain,-1]
    test=data[lentrain-1:,0:4]
    test_labels=data[lentrain-1:,-1]
    
    
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)
	
	
	
def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

	
    # TODO : le code ici pour lire le dataset
    
    lendata=sum(1 for line in f)
    f.close()
    f = open('datasets/binary-winequality-white.csv', 'r')
    data=np.zeros((lendata,12))
    i=0
    for line in f:
        listData,lenList=getDataFromLine(line)
        if lenList != 12:
            print("erreur longueur list mauvaise :",lenList,"\n")
        acdFix=float(listData[0])
        acdVol=float(listData[1])
        acdCtr=float(listData[2])
        sucrRes=float(listData[3])
        chlSod=float(listData[4])
        dioxSoufLib=float(listData[5])
        dioxSoufTot=float(listData[6])
        dens=float(listData[7])
        pH=float(listData[8])
        sulfPot=float(listData[9])
        alcool=float(listData[10])
        bon=int(listData[11])

        data[i,0]=acdFix
        data[i,1]=acdVol
        data[i,2]=acdCtr
        data[i,3]=sucrRes
        data[i,4]=chlSod
        data[i,5]=dioxSoufLib
        data[i,6]=dioxSoufTot
        data[i,7]=dens
        data[i,8]=pH
        data[i,9]=sulfPot
        data[i,10]=alcool
        data[i,11]=bon
        i +=1
    f.close()
    np.random.shuffle(data)
    lentrain=int(train_ratio*lendata)
    train=data[0:lentrain,0:11]
    train_labels=data[0:lentrain,-1]
    test=data[lentrain-1:,0:11]
    test_labels=data[lentrain-1:,-1]

	# La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    conversion_sexe = {'M': 0, 'F' : 1, 'I' : 2}

    f = open('datasets/abalone-intervalles.csv', 'r') # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.

    lendata=sum(1 for line in f)
    f.close()
    f = open('datasets/abalone-intervalles.csv', 'r')
    data=np.zeros((lendata,9))
    i=0
    for line in f:
        listData,lenList=getDataFromLine(line)
        if lenList != 9:
            print("erreur longueur list mauvaise :",lenList,"\n")
        sexe=conversion_sexe[listData[0]]
        longCoq=float(listData[1])
        dimCoq=float(listData[2])
        hauteur=float(listData[3])
        poidTot=float(listData[4])
        poidChair=float(listData[5])
        poidVisc=float(listData[6])
        poidCoq=float(listData[7])
        interNbrAnneaux=int(float(listData[8]))

        data[i,0]=sexe
        data[i,1]=longCoq
        data[i,2]=dimCoq
        data[i,3]=hauteur
        data[i,4]=poidTot
        data[i,5]=poidChair
        data[i,6]=poidVisc
        data[i,7]=poidCoq
        data[i,8]=interNbrAnneaux
        i +=1
    f.close()
    np.random.shuffle(data)
    lentrain=int(train_ratio*lendata)
    train=data[0:lentrain,0:9]
    train_labels=data[0:lentrain,-1]
    test=data[lentrain-1:,0:9]
    test_labels=data[lentrain-1:,-1]

    return (train, train_labels, test, test_labels)