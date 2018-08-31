import sys
import collections
import sys
import random
import numpy as np
from collections import OrderedDict
from collections import defaultdict
import math
def sgn(scalar):
	return 1 if scalar>=0 else -1


#ACCURACY
def accuracy(prdy, outy, count1):
	d=0; e=0
	for i in range(count1):
		l=prdy[i]+1
		if(l==outy[i]):
			d=d+1
		else:
			e=e+1
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

#LOGISTIC REGRESSION
def logistic(data,label, weight, b, lr,t, mar, upd):
	l=[]; inp=[]; prediction=[]; 
	#random.shuffle(data)
	weightvector_input = np.zeros([len(data)])
	for k,v in data.items():
		result={}
		for key in v.keys():
			#print v.get(key)
			result[key] =int(v.get(key,0)) * float(weight.get(key,0))
		summation =sum(map(float,result.values()))
		final=summation +b
		weightvector_input[int(k)]=final
		#print final
		fn1= int(label[int(k)]) * final
		prediction.append(sgn(final))
		if(upd!=0):
			if(fn1<=1):
				for key, value in v.items():
					numerator = (int(label[int(k)])*int(value))
					denominator = 1+math.exp(fn1)
					first_v = numerator/denominator			
					temp = (2*weight[key])/mar
					weight[key] = weight[key] -lr *(-(first_v) + temp)
				b=b+(lr/denominator)
			else:
				for key, value in v.items():
						weight[key] =(1 - lr) * weight[key]
				b= (1-lr) *b
	return weight, b, prediction, weightvector_input

#PREDICT logistic regression
def predict_logistic(dataset, trainlabel, testdata, testlabel, bestlr, bestmargin):
	
	epoch=10
	ll=[]; ll1=[]; allaccuracies=[]; max1=0
	t=0
	global update12
	weight = defaultdict(lambda :random.uniform(-0.01,0.01))
	b=random.uniform(-0.01,0.01)
	update12=0
	accurate=0; max123=0; ll=[]
	updated_weight, bias, nl, weightinptrain = logistic(dataset, trainlabel, weight, b , bestlr,t,bestmargin, 1)
	weight23, bias23, predlabel, weightvector_input = logistic(testdata,testlabel, updated_weight, bias, bestlr,t,bestmargin, 0)
	print "Best learning rate: %0.4f" % bestlr
	print "Best margin: %0.4f" % bestmargin
	#print "\n"
	return	weightvector_input

#CROSS VALIDATION LOGISTIC
def cross_validation_logistic(classifier):
	splitdata={0,1,2,3,4}
	learning_rate=[1,0.1,0.01]; mydict={}
	epoch=1
	maxlr=0
	margin=[0.1,1,10]
	print "logistic regression"
	for lr in learning_rate:
		for m in margin:
			accuracies=[]; 
			for index in splitdata:
        			validlabel, validata = read_data('Data{}/split0{}.txt'.format(classifier,index+1))
				weight = defaultdict(lambda :random.uniform(-0.01,0.01))
				b=random.uniform(-0.01,0.01)
				data=[]
				t=0
				#print index
				for id2 in splitdata - {index}:
					with open("Data{}/split0{}.txt".format(classifier,id2+1),"r") as f1:
						data1= f1.readlines()
						data=data+data1	
				traindatalabel,traindatatot = dataset_crv(data)		
				for i in range(epoch):
				#print len(traindata)	
					crossweight,vf,rt, weightvectorinput =logistic(traindatatot,traindatalabel, weight, b,lr,t,m,1)
				updateweight,b12, predlabel, weightvector_input_test = logistic(validata, validlabel, crossweight, vf, lr,t,m, 0)
				lengthvalidate=len(validata)
				crossaccur = accuracy_crv(predlabel, validlabel, lengthvalidate)
				accuracies.append(crossaccur)
			cal=0.0	
			for i in accuracies:
				cal=cal+ i
			average1=float(cal)/len(splitdata)
			mydict[(lr,m)] = average1
			#print max(mydict.items(), key = lambda k:k[1])
			#print average1
			if(average1>maxlr):
				maxlr=average1
				bestlrcv=lr
				bestmcv=m
	print "Logistic regression:"
	print "Maximum accuracy(crv): %0.4f " % maxlr
	print "learning rate: %0.4f" % bestlrcv
	print "loss tradeoff: %0.4f" % bestmcv

	return bestlrcv, bestmcv



#MAIN
'''label, nestdict = read_data(sys.argv[1])
testlabel, testdict =read_data(sys.argv[2])
predict(nestdict, label, testdict, testlabel, 1,0.1)'''
#bestlr, bestmcv=cross_validation()
testlabel, testdict =read_data('featuretest.txt')
weight_inp_list=np.empty([len(testlabel),6])
split_classifier={1,2,3,4,5,6}
index=0
for i in split_classifier:
	label, nestdict = read_data('split{}.txt'.format(i))
	bestlr, bestmcv=cross_validation_logistic(i)
	weight_input = predict_logistic(nestdict, label, testdict, testlabel, bestlr,bestmcv)
	weight_inp_list[:,index]=weight_input
	index=index+1
#print weight_inp_list.shape
final_argmax = np.argmax(weight_inp_list, axis=1)
#print final_argmax.shape
final_argmax_list = list(final_argmax)

#print testlabel
final_accuracy = accuracy(final_argmax_list, testlabel, len(testdict))
print final_accuracy






