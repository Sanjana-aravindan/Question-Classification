import spacy
import sys
import re
import numpy as np
from spacy.symbols import nsubj, attr, NOUN, PROPN, nsubjpass
from nltk.tree import Tree
from nltk.corpus import treebank
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as p
np.set_printoptions(precision=6)
import scipy.io as a
import codecs
data=[]
data2=[]
f2=open("fv.txt","w")
#with open("trainset.txt","r") as myfile:
stopwords = stopwords.words('english')
en_nlp=spacy.load("en")

def multiclass_split(feature_vector):
	#print feature_vector['ABBR']
	#classlabels=['ABBR', 'DESC', 'HUM','ENTY','NUM','LOC']
	classlabels = ['1','2','3','4','5','6']
	for i in classlabels:
		f3=open("feature_wh_head_train.txt","r")	
		output=open('splitheadwh{}.txt'.format(i),'w')
		for j in f3:
			line=j.strip()
			print line
			first_label=line.split()
			#print first_label[0]
			if first_label[0] == i:
				output.write('1' + " " + line[2:])
			else:
				output.write('-1' + " " +line[2:])
			output.write("\n")
		output.close()	
		f3.close()


def wh_head_words(finalsentence,word_vector,length,label,flag,f2):	
	label_vector ={}
	label_vector['ABBR']=1
	label_vector['DESC']=2
	label_vector['HUM']=3
	label_vector['ENTY']=4
	label_vector['NUM']=5
	label_vector['LOC']=6
	
	whwords=['what','who','why','when','where','how','how many','which','how much']
	if (flag==1):
		featurevector=np.zeros([5452,length])
	else:
		featurevector=np.zeros([500,length])
	j=0
	head_word="null"
	final_vector={}
	for i in finalsentence:
		lab_temp=label_vector[label[j]]
		f2.write("%d" % lab_temp + " ")
		hword_list=[]
		en_doc=en_nlp(u'' + i.lower())
		#print en_doc
		nl=en_nlp(u'' +'how')
		for sent in en_doc.sents:
			if(str(sent[0])=="how" and (str(sent[1])=="many" or str(sent[1])=="much")):
				st=str(sent[0:2]).lower()
				#print st
			else:
				st=str(sent[0]).lower()	
			if(st in whwords):
				#print st
				if (len(st)>5):
					first,second=st.split(" ")
					s=word_vector[first]
					s1=word_vector[first]
					hword_list.append(s)
					hword_list.append(s1)
				else:
					s=word_vector[st]
					hword_list.append(s)
			#st=str(sent[0])
			#featurevector[j][s]=1
			#print sent[0]
			for token in sent:
				#print token
				#print token, token.dep_
				#token.pos_
				#if (token.pos==NOUN or token.pos==PROPN):
					#print token,token.dep_

				if token.dep== nsubj and(token.pos==NOUN or token.pos==PROPN):
					head_word=token.text
				elif token.dep== attr and (token.pos==NOUN or token.pos==PROPN):
					head_word=token.text
				elif token.dep==nsubjpass and (token.pos==NOUN or token.pos==PROPN):
					head_word=token.text
				#elif token.dep==nsubjpass and token.pos==VERB or 
			#print (i+" ("+head_word+")")		
			if head_word in word_vector.keys():
				k=word_vector[head_word]
				hword_list.append(k)
			hword_list.sort()
			for i in hword_list:
				f2.write("%d:1" % i  + " ")
		f2.write("\n")

		j=j+1
			
	
	#print featurevector
	#print np.shape(featurevector)
			
	

def distinctwords(finalsentence):
	finalwordset=[]; wordlist=[]; distinct_word_dict={}
	for s in finalsentence:
		words=word_tokenize(s)
		finalwordset.append(words)
	for i in finalwordset:
		for j in i:
			wordlist.append(j.lower())
	#print wordlist
	wordstok=filter(str.isalnum,wordlist)
	distinctwords1=set(wordstok)
	c=1
	for i in distinctwords1:
		distinct_word_dict[i]=c
		c=c+1
	print c	
	return distinctwords1, distinct_word_dict

				
	

def train(f3,flag):
	label=[]; finalsentence=[]; trainset=[]
	with open(f3,"r") as f1:
		for i in f1:
			#print i
			lab,sent=i.split(":",1)
			#print sent
			finalsent = sent.split(' ',1)[1]
			label.append(lab)  #labels
			finalsentence.append(finalsent.strip()) #sentences
			trainset.append(lab+":"+ finalsent.strip()) #label and sentences		
		ds,distinct_word_vec=distinctwords(finalsentence)
		length=len(distinct_word_vec)
		f1=open("feature_wh_head_train.txt","w")
		wh_head_words(finalsentence,distinct_word_vec,length,label,flag,f1)
		f1.close()
		multiclass_split(distinct_word_vec)
	return distinct_word_vec


def test(f3,distinct_word_vec,flag):
	label=[]; finalsentence=[]; trainset=[]
	with open(f3,"r") as f1:
		for i in f1:
			#print i
			lab,sent=i.split(":",1)
			#print sent
			finalsent = sent.split(' ',1)[1]
			label.append(lab)  #labels
			finalsentence.append(finalsent.strip()) #sentences
			trainset.append(lab+":"+ finalsent.strip()) #label and sentences		
		length=len(distinct_word_vec)
		f2=open("feature_wh_head_test.txt","w")
		wh_head_words(finalsentence,distinct_word_vec,length,label,flag,f2)
		f2.close()
dis_word_vec = train(sys.argv[1],1)
test(sys.argv[2],dis_word_vec,0)
f2.close()
