import sys
import numpy as np
import re
from nltk.tokenize import word_tokenize

def testfeaturevector(feature_vector,trainset,f2):
	label_vector ={}
	label_vector['ABBR']=1
	label_vector['DESC']=2
	label_vector['HUM']=3
	label_vector['ENTY']=4
	label_vector['NUM']=5
	label_vector['LOC']=6
	#print label_vector
	#print trainset
	for s in trainset:
		featurevector=[]	
		sentence=[]; wordstok=[]; 
		label, sent=s.split(":",1)
		sent=sent.lower()
		label_v=label_vector[label]
		sent1=re.sub('[\",\',`,\\n]','',sent)
		words=word_tokenize(sent1)
		#wordstok.append(re.sub('[.,\',\",`,\\n]','',sent))
		wordstok=words
		label_v = label_vector[label]
		featurevector.append(label_v)
		#print featurevector
		for j in range(len(wordstok)-2):
			if(wordstok[j+2]!='?'):
				t=wordstok[j] + " " + wordstok[j+1] + " " + wordstok[j+2]
				if t in feature_vector.keys():
					k=feature_vector[t]
					featurevector.append(k)
				else:
					k=feature_vector['UNK']
					#featurevector.append(k)
		#print feature_vector
		f2.write("%d" % label_v + " ")
		featurevector.pop(0)
		featurevector.sort()
		for i in featurevector:
			f2.write("%d:1" % i  + " ")
		f2.write("\n")


def trigram(trainset):
	i=0

	bigrams=[]
	#print type(featurevector)
	c=1; class_vector={}
	for s in trainset:	
		sentence=[]; wordstok=[]; 
		label, sent=s.split(":",1)
		#print label
		sent=sent.lower()
		sent1=re.sub('[\",\',`,\\n]','',sent)
		#print sent
		words=word_tokenize(sent1)
		#wordstok.append(re.sub('[.,\',\",`,\\n]','',sent))
		wordstok=words
		#wordstok = filter(str.isalnum,words)
		for j in range(len(wordstok)-2):
			flag=1
			if(wordstok[j+2]!='?'):
				d= wordstok[j] + " " + wordstok[j+1] + " " + wordstok[j+2]
				flag=0
				bigrams.append(d)
				if d not in class_vector.keys():
					class_vector[d]=c
					c=c+1
		
	#print bigrams
	distinctbigrams=set(bigrams)
	return distinctbigrams, class_vector,c	

def multiclass_split(feature_vector):
	#print feature_vector['ABBR']
	#classlabels=['ABBR', 'DESC', 'HUM','ENTY','NUM','LOC']
	classlabels = ['1','2','3','4','5','6']
	for i in classlabels:
		f3=open("feature_trigram_train.txt","r")	
		output=open('splittrigram{}.txt'.format(i),'w')
		for j in f3:
			line=j.strip()
			first_label=line.split()
			#print first_label[0]
			if first_label[0] == i:
				output.write('1' + " " + line[2:])
			else:
				output.write('-1' + " " +line[2:])
			output.write("\n")
		output.close()	
		f3.close()
				
def trigram_feature(distinctbigrams, trainset,length,class_vector, f2):

	label_vector ={}
	label_vector['ABBR']=1
	label_vector['DESC']=2
	label_vector['HUM']=3
	label_vector['ENTY']=4
	label_vector['NUM']=5
	label_vector['LOC']=6
	for s in trainset:
		featurevector=[]	
		sentence=[]; wordstok=[]; 
		label, sent=s.split(":",1)
		sent=sent.lower()
		label_v=label_vector[label]
		sent1=re.sub('[\",\',`,\\n]','',sent)
		words=word_tokenize(sent1)
		#wordstok.append(re.sub('[.,\',\",`,\\n]','',sent))
		wordstok=words
		label_v = label_vector[label]
		featurevector.append(label_v)
		#print featurevector
		for j in range(len(wordstok)-2):
			if(wordstok[j+2]!='?'):
				t=wordstok[j] + " " + wordstok[j+1] + " " + wordstok[j+2]
				#print t
				if t in class_vector.keys():
					k=class_vector[t]
					#print k
					featurevector.append(k)
		f2.write("%d" % label_v + " ")
		featurevector.pop(0)
		featurevector.sort()
		for i in featurevector:
			f2.write("%d:1" % i  + " ")
		f2.write("\n")
			
#main training
def train(f2):
	


	label=[]; finalsentence=[]; trainset=[]
	with open(f2,"r") as f1:
			for i in f1:
				lab,sent=i.split(":",1)
				#print sent
				finalsent = sent.split(' ',1)[1]
				label.append(lab)  #labels
				finalsentence.append(finalsent.strip()) #sentences
				trainset.append(lab+":"+ finalsent.strip()) #label and sentences
	#ds=distinctwords(finalsentence)
	distincttrigrams, class_vector,c_count = trigram(trainset)
	#print class_vector
	class_vector['UNK']=c_count+1
	#print distincttrigrams
	f1=open("feature_trigram_train.txt","w")
	lengthbigrams=len(distincttrigrams)
	trigram_feature(distincttrigrams,trainset,lengthbigrams,class_vector,f1)
	f1.close()
	multiclass_split(class_vector)
	#bagofwords(finalsentence) #bag of words - 1st feature
	#predict(dd)
	return class_vector, distincttrigrams

def test(f2, feature_vector):
	with open(f2,"r") as f1:
		#print f1.read()
		label=[]; finalsentence=[]; trainset=[]
		for i in f1:
			#print i
			lab,sent=i.split(":",1)
			#print sent
			finalsent = sent.split(' ',1)[1]
			label.append(lab)  #labels
			finalsentence.append(finalsent.strip()) #sentences
			trainset.append(lab+":"+ finalsent.strip()) #label and sentences
	#print feature_vector['UNK']
	#print trainset
	f2=open("feature_trigram_test.txt","w")
	testfeaturevector(feature_vector, trainset,f2)
	f2.close()


#main
cl_vector, dis_bigrams = train(sys.argv[1])
test(sys.argv[2], cl_vector)

