Programming language: Python 2.7

FEATAURE EXTRACTION:

7 features were extracted from the TREC dataset ( train dataset - 5452 labeled questions & test dataset - 500 questions):

1.  Bag of words (Unigrams) - unigram_feat.py

2.  Bigrams  - bigram_feat.py

3.  Trigrams  -  trigram_feat.py

4.  Unigrams + bigrams  -  unigram-bigram-feat.py

5.  Head words  - head-word.py

6.  Wh words + head words   -   wh-head-feat.py

7.  Word shapes  -  word-shape.py

The extraction of these 7 features are present in the feature directory.


ALGORITHMS:

5 algorithms were implemented on each of these features and comparison of the accuracies were made.

The 5 machine learning algorithms used are: 

1.  Perceptron - perceptron.py

2.  Logistic Regression  -  logistic.py

3.  Support Vector Machine  -  svm.py

4.  Naive Bayes  -  naiave_bayes.py

5.  K- Nearest Neighbor  - knnalgo.py


The above 5 algorithm python files are stored in the Algorithms directory.


Libraries used:  numpy, scipy


