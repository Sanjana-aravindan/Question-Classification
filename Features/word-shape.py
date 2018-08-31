import sys
import numpy as np
from nltk.tokenize import word_tokenize

def wordshapes(finalsentence, label, distinct_word_dict, distinct_words, f2):
	label_vector ={}
	label_vector['ABBR']=1
	label_vector['DESC']=2
	label_vector['HUM']=3
	label_vector['ENTY']=4
	label_vector['NUM']=5
	label_vector['LOC']=6
	bigrams=[]; 	
	#featurevector=np.zeros([5452,2])
	#print type(featurevector)
	lb=0
	for s in finalsentence:	
		featurevector=[]
		sentence=[]; wordstok=[]; 
		words=word_tokenize(s)
		l=label[lb]
		label_v = label_vector[str(l)]
		#wordstok.append(re.sub('[.,\',\",`,\\n]','',sent))
		wordstok=words
		#wordstok = filter(str.isalnum,words)
		for j in wordstok:
			if j.isupper():
				if j in distinct_words:
					w1=distinct_word_dict[j]
					featurevector.append(w1)
			if j[0].isupper():
				if j in distinct_words:
					w1=distinct_word_dict[j]
					if w1 not in featurevector:
						featurevector.append(w1)
		f2.write("%d" % label_v + " ")
		featurevector.sort()
		#print finalvector
		for i in featurevector:
			f2.write("%d:1" % i + " ")
		f2.write('\n')
		lb=lb+1
	#print featurevector
	return featurevector

def multiclass_split(feature_vector):
	#print feature_vector['ABBR']
	#classlabels=['ABBR', 'DESC', 'HUM','ENTY','NUM','LOC']
	classlabels = ['1','2','3','4','5','6']
	for i in classlabels:
		f3=open("featuretrain.txt","r")	
		output=open('split_wordshape{}.txt'.format(i),'w')
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

def distinctwords(finalsentence):
	finalwordset=[]; wordlist=[]; distinct_word_dict={}
	for s in finalsentence:
		words=word_tokenize(s)
		finalwordset.append(words)
	for i in finalwordset:
		for j in i:
			wordlist.append(j)
	#print wordlist
	#wordstok=filter(str.isalnum,wordlist)
	distinctwords1=set(wordlist)
	c=1
	for i in distinctwords1:
		distinct_word_dict[i]=c
		c=c+1
	#print distinct_word_dict
	return distinct_word_dict, c, distinctwords1
	#stopped_words = [i for i in distinctwords if not i in stopwords]
	#print stopped_words


def train(f2):
	label=[]; finalsentence=[]; trainset=[]
	with open(f2,"r") as f1:
		for i in f1:
			lab,sent=i.split(":",1)
			finalsent = sent.split(' ',1)[1]
			label.append(lab)  #labels
			finalsentence.append(finalsent.strip()) #sentences
			trainset.append(lab+":"+ finalsent.strip()) #label and sentences
	distinct_word_dict, c, distinctwords1 = distinctwords(finalsentence)
	distinct_word_dict['UNK']=c
	f1=open("featuretrain.txt","w")
	wordshapes(finalsentence, label, distinct_word_dict, distinctwords1, f1)
	f1.close()
	multiclass_split(distinct_word_dict)
	return distinct_word_dict, distinctwords1

def test(f2, feature_vector, distinct_words):
	with open(f2,"r") as f1:
		label=[]; finalsentence=[]; trainset=[]
		for i in f1:
			lab,sent=i.split(":",1)
			#print sent
			finalsent = sent.split(' ',1)[1]
			label.append(lab)  #labels
			finalsentence.append(finalsent.strip()) #sentences
			trainset.append(lab+":"+ finalsent.strip()) #label and sentences
	print feature_vector['UNK']
	f2=open("featuretest.txt","w")
	wordshapes(finalsentence,label, feature_vector, distinct_words, f2)
	f2.close()

final_vector, distinct_words = train(sys.argv[1])
test(sys.argv[2], final_vector, distinct_words)
