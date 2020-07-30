#!/usr/bin/env python
# coding: utf-8

# In[6]:


import json
import pandas as pd
import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import preprocessing
from sklearn import linear_model
from joblib import dump, load 
import numpy as np
nltk.download('wordnet')
nltk.download('punkt')


# In[ ]:
def process(x):
  return " " .join([stemmer.stem(lemmatizer.lemmatize(i)) for i in set(nltk.word_tokenize(x))])

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    model=classifier.fit(feature_vector_train, label)
    dump(model,'LRMODELGOOD')    
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)


# In[8]:
if __name__== '__main__':
	valid=pd.read_json('snli_1.0/snli_1.0_dev.jsonl',encoding='utf-8',lines=True)
	train=pd.read_json('snli_1.0/snli_1.0_train.jsonl',encoding='utf-8',lines=True)
	test=pd.read_json('snli_1.0/snli_1.0_test.jsonl',encoding='utf-8',lines=True)

	train['annotator_labels']=train['annotator_labels'].apply(lambda text: " ".join(text))
	valid['annotator_labels']=valid['annotator_labels'].apply(lambda text: " ".join(text))
	test['annotator_labels']=test['annotator_labels'].apply(lambda text: " ".join(text))
	train.drop_duplicates()
	valid.drop_duplicates()
	test.drop_duplicates()
	train.query('gold_label!="-"',inplace= True)
	valid.query('gold_label!="-"',inplace= True)
	test.query('gold_label!="-"',inplace= True)
	train['sentence']=train['sentence1'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))+"\n"+train['sentence2'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))
	valid['sentence']=valid['sentence1'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))+"\n"+valid['sentence2'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))
	test['sentence']=test['sentence1'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))+"\n"+test['sentence2'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))
	train=train.drop(columns=['annotator_labels','captionID','pairID','sentence1_binary_parse','sentence1_parse','sentence2_binary_parse','sentence2_parse','sentence1','sentence2'])
	valid=valid.drop(columns=['annotator_labels','captionID','pairID','sentence1_binary_parse','sentence1_parse','sentence2_binary_parse','sentence2_parse','sentence1','sentence2'])
	test=test.drop(columns=['annotator_labels','captionID','pairID','sentence1_binary_parse','sentence1_parse','sentence2_binary_parse','sentence2_parse'  ,'sentence1','sentence2'])
	
	lemmatizer = WordNetLemmatizer() 
	stemmer = SnowballStemmer(language='english')
	

	
	train['sentence']=train['sentence'].apply(lambda x:process(x))
	test['sentence']=test['sentence'].apply(lambda x:  process(x))
	valid['sentence']=valid['sentence'].apply(lambda x:process(x))
	
	print(train.head())
	print(test.head())
	print(valid.head())
	
	encoder = preprocessing.LabelEncoder()
	train_y = encoder.fit_transform(train['gold_label'])
	valid_y = encoder.fit_transform(valid['gold_label'])
	test_y = encoder.fit_transform(test['gold_label'])
	
	list(encoder.classes_)
	
	tfidf_vect = TfidfVectorizer(analyzer='word',max_features=15000) 
	tfidf_vect=tfidf_vect.fit(train['sentence'])
	
	dump(tfidf_vect,'vectorizer')
	
	xtrain_tfidf = tfidf_vect.transform(train['sentence'])
	xvalid_tfidf = tfidf_vect.transform(valid['sentence'])
	xtest_tfidf =  tfidf_vect.transform(test['sentence'])
	

	
	accuracy = train_model(linear_model.LogisticRegression(max_iter=10000), xtrain_tfidf, train_y, xvalid_tfidf)
	print ('LR,Accuracy : %f ' %accuracy )
	
	classifier1 = SGDClassifier(loss='log', alpha=0.0001, penalty='l2')
	model2=classifier1.fit(xtrain_tfidf, train_y)
	predictions = classifier1.predict(xvalid_tfidf)
	dump(model2, 'LRMODEL')  
	print("accuracy :",metrics.accuracy_score(valid_y,predictions))





# In[ ]:




