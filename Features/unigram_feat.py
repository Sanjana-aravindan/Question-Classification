import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
		words=word_tokenize(sent)
		label_v = label_vector[label]
		finalvector.append(label_v)
		wordstok = filter(str.isalnum,words)
		for w in wordstok:
			if w in distinct_words_list:
				w1=feature_vector[w]
				finalvector.append(w1)
			else:
				w1=feature_vector['UNK']
				finalvector.append(w1)
		f2.write("%d" % label_v + " ")
		finalvector.pop(0)
		finalvector.sort()
		for i in finalvector:
			f2.write("%d:1" % i + " ")
		f2.write('\n')




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
		words=word_tokenize(sent)
		wordstok = set(words)
		label_v = label_vector[label]
		finalvector.append(label_v)
		for w in wordstok:
			w1=feature_vector[w]
			finalvector.append(w1)
		f2.write("%d" % label_v + " ")
		finalvector.pop(0)
		finalvector.sort()
		for i in finalvector:
			f2.write("%d:1" % i + " ")
		f2.write('\n')

		
#6 DATASETS FOR 6 CLASSIFIERS (TRAIN)	
def multiclass_split():
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
			
		



#DISTINCT WORDS
def distinctwords(finalsentence):
	finalwordset=[]; wordlist=[]; distinct_word_dict={}
	for s in finalsentence:
		words=word_tokenize(s)
		finalwordset.append(words)
	for i in finalwordset:
		for j in i:
			wordlist.append(j)
	distinctwords1=set(wordlist)
	c=1
	for i in distinctwords1:
		distinct_word_dict[i]=c
		c=c+1
	return distinct_word_dict, c, distinctwords1

#main -TRAIN
def train(f2):
	label=[]; finalsentence=[]; trainset=[]
	with open(f2,"r") as f1:
		for i in f1:
			lab,sent=i.split(":",1)
			finalsent = sent.split(' ',1)[1]
			label.append(lab)  #labels
			finalsentence.append(finalsent.strip()) #sentences
			trainset.append(lab+":"+ finalsent.strip()) #label and sentences
	feature_vector, countvalue, ds=distinctwords(finalsentence)    #distinct words
	feature_vector['UNK']=countvalue
	countvalue+=1
	f2=open("featuretrain.txt","w")
	trainfeaturevector(feature_vector, trainset,f2)
	f2.close()
	multiclass_split()
	return feature_vector, ds

#MAIN - TEST
def test(f2, feature_vector):
	with open(f2,"r") as f1:
		label=[]; finalsentence=[]; trainset=[]
		for i in f1:
			lab,sent=i.split(":",1)
			finalsent = sent.split(' ',1)[1]
			label.append(lab)  #labels
			finalsentence.append(finalsent.strip()) #sentences
			trainset.append(lab+":"+ finalsent.strip()) #label and sentences
	print feature_vector['UNK']
	f2=open("featuretest.txt","w")
	testfeaturevector(feature_vector, trainset,f2)
	f2.close()



#MAIN
fv, distinct_words_list =train(sys.argv[1])
test(sys.argv[2], fv)





	
