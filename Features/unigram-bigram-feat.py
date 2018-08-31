import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

stopwords = stopwords.words('english')

#test set final vector written to  file
def testfeaturevector(feature_vector,trainset,f2):
	label_vector ={}
	label_vector['ABBR']=1
	label_vector['DESC']=2
	label_vector['HUM']=3
	label_vector['ENTY']=4
	label_vector['NUM']=5
	label_vector['LOC']=6
	for s in trainset:	
		finalvector=[]; sentence=[]
		label, sent=s.split(":",1)
		#print label
		sent1=re.sub('[\",\',`,\\n]','',sent)
		words=word_tokenize(sent1)
		label_v = label_vector[label]
		#lab=feature_vector[label]
		finalvector.append(label_v)
		wordstok = filter(str.isalnum,words)
		for w in range(len(wordstok)-1):
			#print w
			if wordstok[w] in distinct_words_list:
				w1=feature_vector[wordstok[w]]
				finalvector.append(w1)
			else:
				w1=feature_vector['UNK']
			if(wordstok[w]!='?'):
				t=wordstok[w] + " " + wordstok[w+1]
				if t in feature_vector.keys():
					k=feature_vector[t]
					finalvector.append(k)
				else:
					k=feature_vector['UNK']
		f2.write("%d" % label_v + " ")
		finalvector.pop(0)
		finalvector.sort()
		for i in finalvector:
			f2.write("%d:1" % i + " ")
		f2.write('\n')
	#print feature_vector['DESC']




def multiclass_split(feature_vector):
	#print feature_vector['ABBR']
	#classlabels=['ABBR', 'DESC', 'HUM','ENTY','NUM','LOC']
	classlabels = ['1','2','3','4','5','6']
	for i in classlabels:
		f3=open("featuretrain.txt","r")	
		output=open('split{}.txt'.format(i),'w')
		for j in f3:
			line=j.strip()
			first_label=line.split()
			print first_label[0]
			if first_label[0] == i:
				output.write('1' + " " + line[2:])
			else:
				output.write('-1' + " " +line[2:])
			output.write("\n")
		output.close()	
		f3.close()




#trainfeature_vector

def trainfeaturevector(feature_vector,trainset,f2):
	label_vector ={}
	label_vector['ABBR']=1
	label_vector['DESC']=2
	label_vector['HUM']=3
	label_vector['ENTY']=4
	label_vector['NUM']=5
	label_vector['LOC']=6
	for s in trainset:	
		finalvector=[]; sentence=[]
		label, sent=s.split(":",1)
		#print label
		#sent1=re.sub('[\",\',`,\\n]','',sent)
		words=word_tokenize(sent)
		wordstok = list(set(words))
		label_v = label_vector[label]
		#print label_v
		#lab=feature_vector[label]
		finalvector.append(label_v)
		for w in range(len(wordstok)-1):
			#print w
			w1=feature_vector[wordstok[w]]
			finalvector.append(w1)
			if(wordstok[w]!='?'):
				t=wordstok[w] + " " + wordstok[w+1]
				if t in feature_vector.keys():
					k=feature_vector[t]
					finalvector.append(k)
		f2.write("%d" % label_v + " ")
		finalvector.pop(0)
		finalvector.sort()
		#print finalvector
		for i in finalvector:
			f2.write("%d:1" % i + " ")
		f2.write('\n')
	#print feature_vector['DESC']
		
	
def multiclass_split(feature_vector):
	#print feature_vector['ABBR']
	#classlabels=['ABBR', 'DESC', 'HUM','ENTY','NUM','LOC']
	classlabels = ['1','2','3','4','5','6']
	for i in classlabels:
		f3=open("featuretrain.txt","r")	
		output=open('split{}.txt'.format(i),'w')
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
			
		

def bigram(trainset, feature_vector):
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
		for j in range(len(wordstok)-1):
			if(wordstok[j+1]!='?'):
				d= wordstok[j] + " " + wordstok[j+1]
				bigrams.append(d)
				if d not in feature_vector.keys():
					feature_vector[d]=c
					c=c+1
	distinctbigrams=set(bigrams)
	return distinctbigrams, feature_vector,c	

#DISTINCT WORDS
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


		


#main
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
	feature_vector, countvalue, ds=distinctwords(finalsentence)    #distinct words
	distinct_bigrams, final_feature_vector, c=bigram(trainset, feature_vector)
	final_feature_vector['UNK']=c
	c+=1
	print c
	#print feature_vector
	f2=open("featuretrain.txt","w")
	trainfeaturevector(final_feature_vector, trainset,f2)
	f2.close()
	multiclass_split(final_feature_vector)
	return final_feature_vector, ds

def test(f2, feature_vector):
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
	testfeaturevector(feature_vector, trainset,f2)
	f2.close()


fv, distinct_words_list =train(sys.argv[1])
test(sys.argv[2], fv)





	
