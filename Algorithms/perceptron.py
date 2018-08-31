import sys
import random
import numpy as np
from collections import defaultdict
from collections import OrderedDict

def sgn(scalar):
	return 1 if scalar>=0 else -1

#SIMPLE PERCEPTRON ALGO
def percep(data,label, weight, b, lr, upd):
	l=[]; inp=[]; pred_y=[]; nl=[];
	
	l=[]; inp=[]; prediction=[]; 
	weightvector_input=np.zeros([len(data)])
	#random.shuffle(data)
	for k,v in data.items():
		result={}
		for key in v.keys():
			#print v.get(key)
			result[key] =int(v.get(key,0)) * float(weight.get(key,0))
		summation =sum(map(float,result.values()))
		#weightvector_input[int(k)]=summation
		final=summation +b
		weightvector_input[int(k)]=final
		#print final
		fn1= int(label[int(k)]) * final
		prediction.append(sgn(final))
		#print weight[1]
		if(upd!=0):
			if(sgn(final)!=int(label[int(k)])):
				for key, value in v.items():
					weight[key] = weight[key] + (lr *float(value) * int(label[int(k)]))
			        b = b + (lr * int(label[int(k)]))	
	#print prediction
	return weight, b, prediction,weightvector_input	


#ACCURACY
def accuracy(prdy, outy, count1):
	d=0; e=0
	for i in range(count1):
		l=prdy[i]+1
		if(l==outy[i]):
			d=d+1
		else:
			e=e+1
	#print count1
	#print d
	acc= (float(d)/count1) * 100
	return acc


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

#SIMPLE PERCEPTRON PREDICT	
def predict(dataset,trainlabel, testdata, testlabel, bestlr):
	
	epoch=10
	predll=[]; ll1=[]; allaccuracies=[]; max1=0
	weight = defaultdict(lambda: random.uniform(-0.01,0.01))
	b=random.uniform(-0.01,0.01)
	#print weight
	accurate=0
	
	updated_weight, bias, nl, weightinptrain = percep(dataset, trainlabel, weight, b , bestlr, 1)
	updated_weight1, bias1, predlabel, weightvector_input = percep(testdata, testlabel, updated_weight, bias , bestlr,0)
	#print "Best accuracy:%0.4f" % max123
	print "Best learning rate: %0.4f" % bestlr
	#print "Total number of updates: %d" % update12
	print  "\n"
	return weightvector_input
		
# SIMPLE PERCEPTRON VALIDATION
def cross_validation(classifier):
	splitdata={0,1,2,3,4}
	learning_rate=[10,1,0.1,0.01]
	epoch=5
	maxlr=0
	for lr in learning_rate:
		accuracies=[]
		for index in splitdata:
        		validlabel, validata = read_data('Data{}/split0{}.txt'.format(classifier,index+1))
			data=[]
			weight = defaultdict(lambda: random.uniform(-0.01,0.01))
			b=random.uniform(-0.01,0.01)
			data=[]
			for id2 in splitdata - {index}:
				with open("Data{}/split0{}.txt".format(classifier,id2+1),"r") as f1:
					data1= f1.readlines()
					data=data+data1	
			traindatalabel,traindatatot = dataset_crv(data)		
			for i in range(epoch):	
				crossweight,vf,rt, weightvectorinput=percep(traindatatot,traindatalabel, weight, b,lr,1)
			updateweight,b12, predlabel, weightvector_input_test = percep(validata, validlabel, crossweight, vf, lr, 0)
			lengthvalidate=len(validata)
			#print validlabel
			crossaccur = accuracy_crv(predlabel, validlabel, lengthvalidate)
			#print crossaccur
			accuracies.append(crossaccur)
		cal=0.0	
		for i in accuracies:
			cal=cal+ i
		average1=float(cal)/len(splitdata)
		if(average1>maxlr):
			#print "hello"
			maxlr=average1
			bestlrcv=lr
	print "Simple perceptron"
	print "Max accuracy(CRV): %0.4f" % maxlr
	print " Best learning rate:%0.4f" % bestlrcv
	return bestlrcv

	  

#MAIN

testlabel, testdict =read_data('featuretest.txt')
weight_inp_list=np.empty([len(testlabel),6])
split_classifier={1,2,3,4,5,6}
index=0
for i in split_classifier:
	label, nestdict = read_data('split{}.txt'.format(i))
	bestlr=cross_validation(i)
	weight_input = predict(nestdict, label, testdict, testlabel, bestlr)
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

