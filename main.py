#!/usr/bin/env python
# coding: utf-8

# In[133]:


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
from sklearn import preprocessing
from sklearn import linear_model
from joblib import dump, load 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import LRModel
import bert
import NNModel
import matplotlib.pyplot as plt 
from sklearn import metrics


nltk.download('wordnet')
nltk.download('punkt')



def process(x):
  return " " .join([stemmer.stem(lemmatizer.lemmatize(i)) for i in set(nltk.word_tokenize(x))])

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    bert_prediction.append(pred_flat)
    bert_label.append(labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


test=pd.read_json('snli_1.0/snli_1.0_test.jsonl',encoding='utf-8',lines=True)
target={'entailment': 0,  'neutral': 1, 'contradiction': 2}
invTarget=['entailment',  'neutral', 'contradiction']


test['annotator_labels']=test['annotator_labels'].apply(lambda text: " ".join(text))
test.drop_duplicates()
test.query('gold_label!="-"',inplace= True)
test['sentence']=test['sentence1'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))+"\n"+test['sentence2'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))
test['bertInput']=test['sentence1'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))+" "+test['sentence2'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))
test['bert_label']=test['gold_label'].apply(lambda x:target[x])
test=test.drop(columns=['annotator_labels','captionID','pairID','sentence1_binary_parse','sentence1_parse','sentence2_binary_parse','sentence2_parse'  ,'sentence1','sentence2'])

bert_x=test['bertInput'].values
bert_y=test['bert_label'].values


lemmatizer = WordNetLemmatizer() 
stemmer = SnowballStemmer(language='english')

test.head()
test['sentence']=test['sentence'].apply(lambda x:process(x))
encoder = preprocessing.LabelEncoder()
test_y = encoder.fit_transform(test['gold_label'])
list(encoder.classes_)

vectorizer = TfidfVectorizer(analyzer='word',max_features=15000) 
vectorizer=load('vectorizer')
xtest_tfidf =  vectorizer.transform(test['sentence'])

LRModel1=linear_model.LogisticRegression
LRModel1=load('Models/LRMODELGOOD')
prediction1=LRModel1.predict(xtest_tfidf)
print("=========================Status of testing Logistics Regression Model ================================")
print('TEST LOSS :',metrics.hamming_loss(prediction1, test_y))
print("TEST ACCURACY :{} %".format(metrics.accuracy_score(prediction1, test_y)*100))

f1=open('Logistics_Regression.txt','w')
f1.write('Loss on Test Data :{}\n'.format(metrics.hamming_loss(prediction1, test_y)))
f1.write('Accuracy on Test Data : {}\n'.format(metrics.accuracy_score(prediction1, test_y)))
f1.write('gt_label,pred_label \n')
for i,j in zip(test_y,prediction1):
    f1.write(str(i))
    f1.write(',')
    f1.write(str(j))
    f1.write("\n")
f1.close()

f1=open('tfidf.txt','w')
for i,j in zip(test_y,prediction1):
    f1.write(invTarget[j])
    f1.write("\n")
f1.close()

LRModel2=SGDClassifier(loss='log', alpha=0.0001, penalty='l2')
LRModel2=load('Models/LRMODEL')
predictions2 = LRModel2.predict(xtest_tfidf)
print ('============================Status of testing Logistics Regression Model using SGD Classifier======================')
print("TEST LOSS :",metrics.hamming_loss(test_y,predictions2))
print("TEST ACCURACY :{} %".format(metrics.accuracy_score(predictions2, test_y)*100))

test_data=np.load('TEST.npy')

TEST=[]
for x,y in zip(test_data,test_y):
  TEST.append([x,y])

test_loader=torch.utils.data.DataLoader(TEST,batch_size=1000,shuffle=True)



net=NNModel.Net().to(device)


net.load_state_dict(torch.load("Models/FeedForward",map_location=device))

net.eval()
correct = 0
total = 0
Vloss=0
prediction2=[]
with torch.no_grad():
        for a in test_loader:
            text, labels = a
            text, labels=text.to(device), labels.to(device)
            outputs = net(text.view(-1,192))
            _, predicted = torch.max(outputs.data, 1)
            for x in predicted:
                prediction2.append(str(int(x)))
            Vloss += F.nll_loss(outputs,labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('========================Status of testing on MultiLayer Neural Network===================================')  
        print('TEST Loss : %f ' %(Vloss.data/len(test_loader)))  
        print('TEST Accuracy : %d '  % (100 * correct / total))

f2=open('multi_layer_net.txt','w')
f2.write('Loss on Test Data :{}\n'.format(Vloss.data/len(test_loader)))
f2.write('Accuracy on Test Data : {}\n'.format(100 * correct / total))
f2.write('gt_label,pred_label \n')
for i,j in zip(test_y,prediction2):
    f2.write(str(i))
    f2.write(',')
    f2.write(str(j))
    f2.write("\n")
f2.close()

input_dim =   NNModel.input_dim 
hidden_dim =  NNModel.hidden_dim 
layer_dim =   NNModel.layer_dim 
output_dim  = NNModel.output_dim 
seq_dim =     NNModel.seq_dim

model = NNModel.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load("Models/LSTM",map_location=device))

correct = 0
total = 0
loss=0
prediction3=[]
for images, labels in test_loader:
    images = images.view(-1, seq_dim, input_dim)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    for x in predicted:
                prediction3.append(str(int(x)))
    total += labels.size(0)
    correct += (predicted == labels).sum()
accuracy = 100 * correct / total
loss=criterion(outputs,labels)
print('========================Status of testing on Recurrent Neural Network===================================')  
print('TEST Loss: {}\nTEST Accuracy: {}%'.format( loss.item(), accuracy))


f3=open('Recur_net.txt','w')
f3.write('Loss on Test Data :{}\n'.format(loss.item()))
f3.write('Accuracy on Test Data : {}\n'.format(100 * correct / total))
f3.write('gt_label,pred_label \n')
for i,j in zip(test_y,prediction3):
    f3.write(str(i))
    f3.write(',')
    f3.write(str(j))
    f3.write("\n")
f3.close()








model = BertForSequenceClassification.from_pretrained('Models/BertModel')
tokenizer =  BertTokenizer.from_pretrained('Models/BertModel')
model.to(device)

bert_prediction=[]
bert_label=[]
input_ids = []
attention_masks = []
for sent in bert_x:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = 128,           
                        pad_to_max_length = True,
                        return_attention_mask = True,  
                        return_tensors = 'pt',     
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(bert_y)
dataset = TensorDataset(input_ids, attention_masks, labels)
batch_size = 8
test_dataloader = DataLoader(
            dataset,  
            sampler = RandomSampler(dataset), 
            batch_size = batch_size 
        )


total_eval_accuracy = 0
total_eval_loss = 0
training_stats=[]
for batch in test_dataloader:
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)
      with torch.no_grad():        
          (loss, logits) = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
          
      total_eval_loss += loss.item()
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      total_eval_accuracy += flat_accuracy(logits, label_ids)
      avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
      avg_val_loss = total_eval_loss / len(test_dataloader)

print('========================Status of testing on Fine Tuned Bert Model===================================')  
print("TEST Loss: {0:.2f}".format(avg_val_loss))
print("TEST Accuracy: {0:.2f}%".format(100*avg_val_accuracy))


f4=open('Bert.txt','w')
f4.write('Loss on Test Data :{}\n'.format(avg_val_loss))
f4.write('Accuracy on Test Data : {}\n'.format(100 * avg_val_accuracy))
f4.write('gt_label,pred_label \n')
for i,j in zip(bert_label,bert_prediction):
    for x,y in zip(i,j):
        f4.write(str(x))
        f4.write(',')
        f4.write(str(y))
        f4.write("\n")
f4.close()

f5=open('deep_model.txt','w')
for i,j in zip(bert_label,bert_prediction):
    for x,y in zip(i,j):
        f5.write(invTarget[y])
        f5.write("\n")
f5.close()


# In[138]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
c_matrix=metrics.confusion_matrix(np.concatenate(bert_prediction,axis=0),np.concatenate( bert_label, axis=0 ))
disp=metrics.ConfusionMatrixDisplay(c_matrix,[0,1,2])
disp.plot(cmap='Blues', ax=ax)
disp.ax_.set_title("Confusion Matrix for Bert Model ")
plt.show()


# In[151]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
c_matrix=metrics.confusion_matrix(np.array(prediction1),np.array(test_y))
disp=metrics.ConfusionMatrixDisplay(c_matrix,[0,1,2])
disp.plot(cmap='Blues', ax=ax)
disp.ax_.set_title("Confusion Matrix for LRModel ")
plt.show()


# In[150]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
c_matrix=metrics.confusion_matrix(np.array([int(x) for x in prediction2]),np.array(test_y))
disp=metrics.ConfusionMatrixDisplay(c_matrix,[0,1,2])
disp.plot(cmap='Blues', ax=ax)
disp.ax_.set_title("Confusion Matrix for multi-layer-net ")
plt.show()


# In[148]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
c_matrix=metrics.confusion_matrix(np.array([int(x) for x in prediction3]),np.array(test_y))
disp=metrics.ConfusionMatrixDisplay(c_matrix,[0,1,2])
disp.plot(cmap='Blues', ax=ax)
disp.ax_.set_title("Confusion Matrix for Recur-neural-net ")
plt.show()


# In[ ]:




