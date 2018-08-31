import sys
import collections
import random
import numpy as np
from numpy import ma
from collections import OrderedDict
import scipy.sparse as sp
import math

#TRAIN DATASET - DENSE MATRIX
def dataset_creation(nestdict,label):
	mat = sp.dok_matrix((len(nestdict), 9440), dtype = np.int8)
	Y=np.empty([len(nestdict)],dtype=int)
	for k, v in nestdict.items():
		for k1 in v.keys():
			mat[int(k),int(k1)]=1
		Y[int(k)]=label[int(k)]
	final_matrix = mat.todense()
	#print final_matrix
	return final_matrix, Y


#GROUND TRUE LABEL COUNT
def label_total_count(trainlabel):
	unique = list(set(trainlabel))
	label_count=[]
	#print j
	count1=count2=count3=count4=count5=count6=0
	for i in trainlabel:
		if i==1:
			count1=count1+1
		elif i==2:
			count2=count2+1
		elif i==3:
			count3=count3+1
		elif i==4:
			count4=count4+1
		elif i==5:
			count5=count5+1
		else:
			count6=count6+1
	label_count=[count1, count2, count3, count4, count5, count6]
	return label_count


#TRAIN - NAIVE BAYES
def naive_bayes(traindata, trainlabel,final_1, final_2, final_3, final_4, final_5, final_6, smooth_term):
	unique_labels = list(set(trainlabel))
	label_count = label_total_count(trainlabel)
	prior_label_prob=[]
	shortmatrix1 = np.empty([len(traindata), 9440])
	shortmatrix2 = np.empty([len(traindata), 9440])
	shortmatrix3 = np.empty([len(traindata), 9440])
	shortmatrix4 = np.empty([len(traindata), 9440])
	shortmatrix5 = np.empty([len(traindata), 9440])
	shortmatrix6 = np.empty([len(traindata), 9440])
	cout1=0; l=[]; 
	'''for i in label_count:
		prior_prob = float(i)/len(traindata)
		prior_label_prob.append(prior_prob)'''
	#prior_neg = float(neg_count)/len(traindata)
	final_matrix,Y = dataset_creation(traindata,trainlabel)
	
	i=0
	for col in trainlabel:
		if col ==1:
			shortmatrix1[i,:] = final_matrix[i]
		elif col==2:
			shortmatrix2[i,:] = final_matrix[i]
		elif col==3:
			shortmatrix3[i,:] = final_matrix[i]
		elif col==4:
			shortmatrix4[i,:] = final_matrix[i]
		elif col==5:
			shortmatrix5[i,:] = final_matrix[i]
		else:
			shortmatrix6[i,:] = final_matrix[i]
		i=i+1
	#excluding the 1st unnecessary column
	final_matrix1=shortmatrix1[:,1:]
	final_matrix2=shortmatrix2[:,1:]
	final_matrix3=shortmatrix3[:,1:]
	final_matrix4=shortmatrix4[:,1:]
	final_matrix5=shortmatrix5[:,1:]
	final_matrix6=shortmatrix6[:,1:]
	
	num1 =(np.sum(final_matrix1, axis=0) + float(smooth_term))
	den1 = (label_count[0] + 2*float(smooth_term)) 
	final_1 = np.divide(num1,den1)

	num2 =(np.sum(final_matrix2, axis=0) + float(smooth_term))
	den2 = (label_count[1] + 2*float(smooth_term)) 
	final_2 = np.divide(num2,den2)

	num3 =(np.sum(final_matrix3, axis=0) + float(smooth_term))
	den3 = (label_count[2] + 2*float(smooth_term)) 
	final_3 = np.divide(num3,den3)

	num4 =(np.sum(final_matrix4, axis=0) + float(smooth_term))
	den4 = (label_count[3] + 2*float(smooth_term)) 
	final_4 = np.divide(num4,den4)
	
	num5 =(np.sum(final_matrix5, axis=0) + float(smooth_term))
	den5 = (label_count[4] + 2*float(smooth_term)) 
	final_5 = np.divide(num5,den5)

	num6 =(np.sum(final_matrix6, axis=0) + float(smooth_term))
	den6 = (label_count[5] + 2*float(smooth_term)) 
	final_6 = np.divide(num6,den6)
	#print label_count[0]
	#print final_1
	
	return final_1, final_2, final_3, final_4, final_5, final_6

#TEST- NAIVE BAYES
def test_naive_bayes(testdata,testlabel,final_1,final_2, final_3, final_4, final_5, final_6):
	
	predlabel=[]; prior_label_prob=[]; 
	label_count = label_total_count(testlabel)
	test_matrix, true_label = dataset_creation(testdata, testlabel)
	for i in label_count:
		prior_prob = float(i)/len(testdata)
		prior_label_prob.append(prior_prob)
	
	testm = sp.dok_matrix((len(testdata),9440), dtype = np.int8)
	testm = test_matrix[:,1:]
	
	
	post_1 = [testm==1]*final_1 + [testm==0] * (1-final_1)
	post_2 = [testm==1]*final_2 + [testm==0] * (1-final_2)
	post_3 = [testm==1]*final_3 + [testm==0] * (1-final_3)
	post_4 = [testm==1]*final_4 + [testm==0] * (1-final_4)
	post_5 = [testm==1]*final_5 + [testm==0] * (1-final_5)
	post_6 = [testm==1]*final_6 + [testm==0] * (1-final_6)
	
	post_1=np.squeeze(post_1,axis=0)
	post_2=np.squeeze(post_2,axis=0)
	post_3=np.squeeze(post_3,axis=0)
	post_4=np.squeeze(post_4,axis=0)
	post_5=np.squeeze(post_5,axis=0)
	post_6=np.squeeze(post_6,axis=0)
	
	res_1 = ma.filled(np.log(ma.masked_equal(post_1,0)),0)
	res_2 = ma.filled(np.log(ma.masked_equal(post_2,0)),0)
	res_3 = ma.filled(np.log(ma.masked_equal(post_3,0)),0)
	res_4 = ma.filled(np.log(ma.masked_equal(post_4,0)),0)
	res_5 = ma.filled(np.log(ma.masked_equal(post_5,0)),0)
	res_6 = ma.filled(np.log(ma.masked_equal(post_6,0)),0)
	
	post_1 = np.sum(res_1,axis=1)
	post_2 = np.sum(res_2,axis=1)
	post_3 = np.sum(res_3,axis=1)
	post_4 = np.sum(res_4,axis=1)
	post_5 = np.sum(res_5,axis=1)
	post_6 = np.sum(res_6,axis=1)
	post_list=[post_1, post_2, post_3, post_4, post_5, post_6]
	
	post_1=post_1 + math.log(prior_label_prob[0])
	post_2=post_2 + math.log(prior_label_prob[1])
	post_3=post_3 + math.log(prior_label_prob[2])
	post_4=post_4 + math.log(prior_label_prob[3])
	post_5=post_5 + math.log(prior_label_prob[4])
	post_6=post_6 + math.log(prior_label_prob[5])

	for row in range(len(testdata)):

		post_final=[post_1[row], post_2[row], post_3[row], post_4[row], post_5[row], post_6[row]]
		predlabel.append(np.argmax(post_final)+1)
	return predlabel
	
#ACCURACY
def accuracy(prdy, outy, count1):
	d=0; e=0
	for i in range(count1):
		if(prdy[i]==outy[i]):
			d=d+1
		else:
			e=e+1
	acc= (float(d)/count1) * 100
	return acc


#RAW DATA PROCESSING
def read_data(filen):
	inputs=[]; i=0; l=0
	label=[]
	listnew=[]; inputs=[]; nesteddict=OrderedDict()
	with open(filen,"r") as f1:
		for line in f1:
			featuredict={}; sorteddict={}; key=[]; value=[]
			w=line.split()
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

#DATASET (DICT)
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



#NAIVE BAYES PREDICT	
def predict(dataset,trainlabel,testdata, testlabel, best):
	epoch=10; max123=0
	predll=[]; ll1=[]; allaccuracies=[]; max1=0
	#print weight
	accurate=0
	final_1 = np.empty([len(dataset), 9440])
	final_2= np.empty([len(dataset),9440])
	final_3 = np.empty([len(dataset), 9440])
	final_4= np.empty([len(dataset),9440])
	final_5 = np.empty([len(dataset), 9440])
	final_6= np.empty([len(dataset),9440])
	final_1, final_2, final_3, final_4, final_5, final_6 = naive_bayes(dataset, trainlabel,final_1, final_2, final_3, final_4, final_5, final_6, best)
	predlabel1 = test_naive_bayes(testdata, testlabel,final_1, final_2, final_3, final_4, final_5, final_6)
	lengthtest1=len(testdata)
	trainaccuracy1=accuracy(predlabel1,testlabel,lengthtest1)
	print "Best accuracy:%0.4f" % trainaccuracy1
	print "Best smoothing term: %0.4f" % float(best)
		
# NAIVE BAYES CROSS VALIDATION
def cross_validation():
	splitdata={0,1,2,3,4}
	smooth =['2','1.5','1.0','0.5']
	epoch=5
	maxlr=0;
	print "naive bayes cross validation"
	for sm in smooth:
		accuracies=[]
		for index in splitdata:
        		validlabel, validata = read_data('CRV/training0{}.txt'.format(index+1))
			data=[]
			for id2 in splitdata - {index}:
				with open("CRV/training0{}.txt".format(id2+1),"r") as f1:
					data1= f1.readlines()
					data=data+data1	
			traindatalabel,traindatatot = dataset_crv(data)	
			final_1 = np.empty([len(traindatatot), 9440])
			final_2= np.empty([len(traindatatot),9440])
			final_3 = np.empty([len(traindatatot), 9440])
			final_4= np.empty([len(traindatatot),9440])
			final_5 = np.empty([len(traindatatot), 9440])
			final_6= np.empty([len(traindatatot),9440])
			for i in range(epoch):	
				final_1, final_2, final_3, final_4, final_5, final_6 = naive_bayes(traindatatot, traindatalabel,final_1, final_2, final_3, final_4, final_5, final_6, sm)	
			predlabel= test_naive_bayes(validata, validlabel,final_1, final_2, final_3, final_4, final_5, final_6)
			#predlabel = test_naive_bayes(validata, validlabel,posmat_updated,negmat_updated)
			#print predlabel
			lengthvalidate=len(validata)
			crossaccur = accuracy(predlabel, validlabel, lengthvalidate)
			accuracies.append(crossaccur)
		cal=0.0	
		for i in accuracies:
			cal=cal+ i
		average1=float(cal)/len(splitdata)
		if(average1>maxlr):
			maxlr=average1
			best_smooth = sm
	print "Naive Bayes: Cross Validation"
	print "Max accuracy(CRV): %0.4f" % maxlr
	print " Best Smoothing term:%0.4f" % float(best_smooth)
	return best_smooth





#MAIN

label, nestdict = read_data(sys.argv[1])
testlabel, testdict =read_data(sys.argv[2])

smooth=cross_validation()
predict(nestdict, label,testdict,testlabel, smooth)



