import numpy as np
import sys
import load_datasets
import Knn # importer la classe du Knn


#nbrEchant : nombre de sous échantillons
nbrEchant = 10
#kMin , kMax : bornes des valeurs possibles de k. Ces valeurs doivent être impairs
kMin=1
kMax=11

listKValues=[kMin]
q=kMin+2 # reste impair
while q < kMax:
	listKValues.append(q)
	q +=2
listKValues.append(kMax)

#tuple=load_datasets.load_iris_dataset(1)
tuple=load_datasets.load_wine_dataset(1)
#tuple=load_datasets.load_abalone_dataset(1)
data=tuple[0]
data_labels=tuple[1]
lendata=len(data)
listEchant=[]
for l in range(0,nbrEchant):
    tempdata = data[ int(l*lendata/(nbrEchant)) : int((l+1)*lendata/(nbrEchant))]
    tempdatalabels = data_labels[int(l*lendata/(nbrEchant)) : int((l+1)*lendata/(nbrEchant))]
    listEchant.append((tempdata,tempdatalabels))

listAccuracy=[]
for k in listKValues:
    moyAccuracy=0
    for L in range(0,nbrEchant):
        test,test_labels=listEchant[L]
        firstEch= True
        for n in range(0,nbrEchant):
            if n != L:
                if firstEch:
                    train=listEchant[n][0]
                    train_labels=listEchant[n][1]
                    firstEch = False
                else:
                    train =np.concatenate((train,listEchant[n][0]),axis=0)
                    train_labels =np.concatenate((train_labels,listEchant[n][1]),axis=0)
        classifieurKnn=Knn.Knn(k)
        classifieurKnn.train(np.array(train),np.array(train_labels))
        accuracy=classifieurKnn.evaluate(np.array(test),np.array(test_labels))[1]
        moyAccuracy += accuracy
    moyAccuracy /= nbrEchant
    listAccuracy.append(moyAccuracy)

maxAccuracy=listAccuracy[0]
maxIndex = 0
for i in range(0,len(listAccuracy)):
    if maxAccuracy < listAccuracy[i]:
        maxAccuracy = listAccuracy[i]
        maxIndex = i

bestKValue = kMin +2*maxIndex

print("La meilleur valeur de k est :",bestKValue)
print(listAccuracy)