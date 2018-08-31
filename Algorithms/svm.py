import sys
import collections
import random
import numpy as np
from collections import OrderedDict
from collections import defaultdict

def sgn(scalar):
	return 1 if scalar>=0 else -1


#ACCURACY
def accuracy(prdy, outy, count1):
	d=0; e=0
	for i in range(count1):
		l=prdy[i]+1
		print l
		if(l==outy[i]):
			d=d+1
		else:
			e=e+1
	acc= (float(d)/count1) * 100
	return acc

#accuracy_cross validation
def accuracy_crv(prdy, outy, count1):
	d=0; e=0
	for i in range(count1):
		if(prdy[i]==outy[i]):
			d=d+1
		else:
			e=e+1
	acc= (float(d)/count1) * 100
	return acc

#RETURNS DICTIONARY OF FEATURES AND LABELS
def read_data(filen):
	inputs=[]; i=0; l=0
	label=[]
	listnew=[]; inputs=[]; nesteddict=OrderedDict()
	with open(filen,"r") as f1:
		for line in f1:
			featuredict={}; sorteddict={}; key=[]; value=[]
			w=line.split()
			#print w
			label.append(int(w[0]))
			w.pop(0)
			#print w
			for j in w:
				x=j.split(":")
				key.append(x[0])
				value.append(x[1])
			featuredict=OrderedDict(zip(key,value))
			#sorteddict=
			#print sorteddict
			if featuredict != {}:
				nesteddict[l]= featuredict
			else:
				nesteddict[l]={}
			l=l+1
	return label, nesteddict

#RETURNS A DICT AND LABEL LIST
def dataset_crv(dataset):
	inputs=[]; i=0; l=0
	label=[]
	listnew=[]; inputs=[]; nesteddict=OrderedDict()
	for line in dataset:
		featuredict={}; sorteddict={}; key=[]; value=[]
		w=line.split()
		#print w
		label.append(int(w[0]))
		w.pop(0)
		#print w
		for j in w:
			x=j.split(":")
			key.append(x[0])
			value.append(x[1])
		featuredict=OrderedDict(zip(key,value))
		if featuredict != {}:
			nesteddict[l]= featuredict
		else:
			nesteddict[l]={}
		l=l+1
	return label, nesteddict

#SVM 
def svm_main(data,label, weight, b, lr,t, mar, upd):
	l=[]; inp=[]; prediction=[]; 
	weightvector_input=np.zeros([len(data)])
	#random.shuffle(data)
	for k,v in data.items():
		result={}
		for key in v.keys():
			result[key] =int(v.get(key,0)) * float(weight.get(key,0))
		summation =sum(map(float,result.values()))
		final=summation +b
		weightvector_input[int(k)]=final
		fn1= int(label[int(k)]) * final
		prediction.append(sgn(final))
		if(upd!=0):
			if(fn1<=1):
				for key, value in v.items():
						weight[key] =((1 - lr) * weight[key]) + (lr * mar *int(value) * int(label[int(k)]))
				b=(((1-lr) *b) + (lr *mar * int(label[int(k)]) *1))
			else:
				for key, value in v.items():
						weight[key] =(1 - lr) * weight[key]
				b= (1-lr) *b
	return weight, b, prediction, weightvector_input


#PREDICT SVM
def predict(dataset, trainlabel,testdata, testlabel, bestlr, bestmargin):
	
	weight = defaultdict(lambda :random.uniform(-0.01,0.01))
	b=random.uniform(-0.01,0.01)
	updated_weight, bias, nl, weightinptrain = svm_main(dataset, trainlabel, weight, b , bestlr,t,bestmargin, 1)
	updated_weight1, bias1, predlabel, weightvector_input = svm_main(testdata, testlabel, updated_weight, bias , bestlr,t,bestmargin, 0)
	print "Best learning rate: %0.4f" % bestlr
	print "	loss tradeoff: %0.4f" % bestmargin
	return weightvector_input

	
#CROSS VALIDATION SVM
def cross_validation(classifier):
	splitdata={0,1,2,3,4}
	epoch=5
	maxlr=0
	learning_rate=[10,1,0.1]
	margin=[10,1,0.1]
	print "CROSS VALIDATION SVM"
	for lr in learning_rate:
		for m in margin:
			accuracies=[]; 
			for index in splitdata:
        			validlabel, validata = read_data('Data{}/split0{}.txt'.format(classifier, index+1))
				weight = defaultdict(lambda :random.uniform(-0.01,0.01))
				b=random.uniform(-0.01,0.01)
				data=[]
				t=0
				for id2 in splitdata - {index}:
					with open("Data{}/split0{}.txt".format(classifier,id2+1),"r") as f1:
						data1= f1.readlines()
						data=data+data1	
				traindatalabel,traindatatot = dataset_crv(data)		
				for i in range(epoch):	
					crossweight,vf,rt, weightvectorinput=svm_main(traindatatot,traindatalabel, weight, b,lr,t,m,1)
				updateweight,b12, predlabel, weightvector_input_test = svm_main(validata, validlabel, crossweight, vf, lr,t,m, 0)
				lengthvalidate=len(validata)
				crossaccur = accuracy_crv(predlabel, validlabel, lengthvalidate)
				accuracies.append(crossaccur)
			cal=0.0	
			for i in accuracies:
				cal=cal+ i
			average1=float(cal)/len(splitdata)
			if(average1>maxlr):
				maxlr=average1
				bestlrcv=lr
				bestmcv=m
	print "SUPPORT VECTOR MACHINE:"
	print "Maximum accuracy(crv): %0.4f " % maxlr
	print "Learning rate: %0.4f" % bestlrcv
	print "lOSS TRADEOFF: %0.4f" % bestmcv

	return bestlrcv, bestmcv




#MAIN
#MULTICLASS CLASSIFIER - ONE VS ALL
testlabel, testdict =read_data('featuretest.txt')
weight_inp_list=np.empty([len(testlabel),6])
split_classifier={1,2,3,4,5,6}
index=0
for i in split_classifier:
	label, nestdict = read_data('split{}.txt'.format(i))
	bestlr, bestmcv=cross_validation(i)
	weight_input = predict(nestdict, label, testdict, testlabel, bestlr,bestmcv)
	weight_inp_list[:,index]=weight_input
	index=index+1
#print weight_inp_list.shape
final_argmax = np.argmax(weight_inp_list, axis=1)
#print final_argmax.shape
final_argmax_list = list(final_argmax)
#print final_argmax_list
#print testlabel
final_accuracy = accuracy(final_argmax_list, testlabel, len(testdict))
print final_accuracy

