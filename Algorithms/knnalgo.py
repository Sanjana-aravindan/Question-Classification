import csv
import random
import math
import operator
import sys

def datasplit(filen):
	dataset=[]; c=0; label=[];
	with open(filen,"r") as f1:
		for line in f1:
			c=c+1
			key=[]; label=[]
			value=[]; listnew=[]; inputs=[]
	
			w=line.split()
			label=float(w[0])
			w.pop(0)
			#k=0
			for j in w:
				x=j.split(":")
				key.append(x[0])
				value.append(float(x[1]))
			inputs=[]
			i=0
			# len(key)
 			for j in range(0,9439):
				if(i<len(key)):
					m=int(key[i])
					if(m==j+1):
						inputs.insert(j+1,value[i])
						i=i+1
					else:
						inputs.insert(j,0)
				else:
					inputs.insert(j,0)	
			inputs.insert(9440,label)
			numbfeat=len(inputs)
			listnew.append(inputs)
			#listnew.insert(9440,label)
			#listnew.append(label)
			dataset.append(inputs)
	return dataset

def test_datasplit(filen):
	dataset=[]; c=0; label=[];
	with open(filen,"r") as f1:
		for line in f1:
			c=c+1
			key=[]; label=[]
			value=[]; listnew=[]; inputs=[]
	
			w=line.split()
			label=float(w[0])
			w.pop(0)
			#k=0
			for j in w:
				x=j.split(":")
				key.append(x[0])
				value.append(float(x[1]))
			inputs=[]
			i=0
			# len(key)
 			for j in range(0,9439):
				if(i<len(key)):
					m=int(key[i])
					if(m==j+1):
						inputs.insert(j+1,value[i])
						i=i+1
					else:
						inputs.insert(j,0)
				else:
					inputs.insert(j,0)	
		
			numbfeat=len(inputs)
			listnew.append(inputs)
			dataset.append(listnew)
		return dataset


def euclideandist(testex1, trainex2, length):
	distance = 0
	for x in range(length):
		dist += pow((testex1[x] - trainex2[x]), 2)
	return math.sqrt(dist)

def neighbors(trainset, testex, k):
	distance = []
	length = len(testex)-1
	for x in range(len(trainset)):
		dist = euclideanDistance(testex, trainset[x], length)
		distance.append((trainset[x], dist))
	distance.sort(key=operator.itemgetter(1))
	neighbors_instances = []
	for x in range(k):
		neighbors_instances.append(distance[x][0])
	return neighbors_instances

def getemajorvotes(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		out = neighbors[x][-1]
		if out in classVotes:
			classVotes[out] += 1
		else:
			classVotes[out] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
#CROSS VALIDATION SVM
def cross_validation():
	splitdata={0,1,2,3,4}
	bestk=0
	print "CROSS VALIDATION K NEAREST NEIGHBORS"
	for lr in k_values:
		accuracies=[]; predictions=[]
		for index in splitdata:
			predictions=[]
        		validlabel, validata = read_data('split0{}.txt'.format(index+1))
			data=[]
			for id2 in splitdata - {index}:
				with open("split0{}.txt".format(id2+1),"r") as f1:
					data1= f1.readlines()
					data=data+data1	
			traindatalabel,traindatatot = dataset_crv(data)		
			for x in range(len(validata)):
				neighbors = neighbors(trainingdatatot, validata[x], lr)
			result = getmajorvotes(neighbors)
			predictions.append(result)
			lengthvalidate=len(validata)
			crossaccur = accuracy_crv(predictions, validlabel, lengthvalidate)
			accuracies.append(crossaccur)
		cal=0.0	
		for i in accuracies:
			cal=cal+ i
		average1=float(cal)/len(splitdata)
		if(average1>maxlr):
			maxlr=average1
			bestk=lr
	print "SUPPORT VECTOR MACHINE:"
	print "best k value: %d " % bestk
	
	return bestk

def main_code():
	# prepare data
	trainingSet=[]
	testSet=[]
	trainingSet = datasplit(sys.argv[1])
	testSet = datasplit(sys.argv[2])
	predictions=[]
	k=cross_validation()
	for x in range(len(testSet)):
		neighbors = neighbors(trainingSet, testSet[x], k)
		result = getmajorvotes(neighbors)
		predictions.append(result)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: %f' % accuracy)
	
main_code()
