import spacy
from spacy.en import English
import sys
import re
import numpy as np
from spacy.symbols import nsubj, attr, NOUN, PROPN, nsubjpass
from nltk.tree import Tree
from nltk.corpus import treebank
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
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
#with open("trainset.txt","r") as myfile:
stopwords = stopwords.words('english')
en_nlp=spacy.load("en")
#nlp=English()
def multiclass_split():
	#print feature_vector['ABBR']
	#classlabels=['ABBR', 'DESC', 'HUM','ENTY','NUM','LOC']
	classlabels = ['1','2','3','4','5','6']
	for i in classlabels:
		f3=open("feature_head_train.txt","r")	
		output=open('splithead{}.txt'.format(i),'w')
		for j in f3:
			line=j.strip()
			#print line
			first_label=line.split()
			#print first_label[0]
			if first_label[0] == i:
				output.write('1' + " " + line[2:])
			else:
				output.write('-1' + " " +line[2:])
			output.write("\n")
		output.close()	
		f3.close()


def head_words(finalsentence, word_vector, length, label, flag, f2):	
	label_vector ={}
	label_vector['ABBR']=1
	label_vector['DESC']=2
	label_vector['HUM']=3
	label_vector['ENTY']=4
	label_vector['NUM']=5
	label_vector['LOC']=6
	if (flag==1):
		featurevector=np.zeros([5452,length])
	else:
		featurevector=np.zeros([500,length])
	j=0
	final_vector={}
	for i in finalsentence:
		head_word="null"
		#print i
		#print label[j]
		lab_temp=label_vector[label[j]]
		f2.write("%d" % lab_temp + " ")
		hword_list=[]
		en_doc=en_nlp(u'' + i.lower())
		#print en_doc
		nl=en_nlp(u'' +'how')
		for sent in en_doc.sents:
			hword_list=[]
			#print sent
			for token in sent:
				if token.dep== nsubj and(token.pos==NOUN or token.pos==PROPN):
					head_word=token.text
				elif token.dep== attr and (token.pos==NOUN or token.pos==PROPN):
					head_word=token.text
				elif token.dep==nsubjpass and (token.pos==NOUN or token.pos==PROPN):
					head_word=token.text	
		if head_word in word_vector.keys():
			#print head_word
			k=word_vector[head_word]
			if k not in hword_list:
				hword_list.append(k)
		#print hword_list
		hword_list.sort()
		for i in hword_list:
			f2.write("%d:1" % i  + " ")
		#print "hello"
		f2.write("\n")
		j=j+1

	'''y=[unicode(i) for i in finalsentence]
	en_doc=nlp(finalsentence)
	#print en_doc
	for sent in en_doc.sents:
		lab_temp=label_vector[label[j]]
		f2.write("%d" % lab_temp + " ")
		hword_list=[]
		for token in sent:
			if token.dep== nsubj and(token.pos==NOUN or token.pos==PROPN):
				head_word=token.text
			elif token.dep== attr and (token.pos==NOUN or token.pos==PROPN):
				head_word=token.text
			elif token.dep==nsubjpass and (token.pos==NOUN or token.pos==PROPN):
				head_word=token.text	
		if head_word in word_vector.keys():
			print head_word
			k=word_vector[head_word]
			if k not in hword_list:
				hword_list.append(k)
		print hword_list
		hword_list.sort()
		for i in hword_list:
			f2.write("%d:1" % i  + " ")
		#print "hello"
		f2.write("\n")'''

			
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
	#print c	
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
		f1=open("feature_head_train.txt","w")
		head_words(finalsentence,distinct_word_vec,length,label,flag,f1)
		f1.close()
		multiclass_split()
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
		f2=open("feature_head_test.txt","w")
		head_words(finalsentence,distinct_word_vec,length,label,flag,f2)
		f2.close()
dis_word_vec = train(sys.argv[1],1)
test(sys.argv[2],dis_word_vec,0)
