
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, random_split
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import string
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__=='__main__':
	if torch.cuda.is_available():    
	    device = torch.device("cuda")
	    print('There are %d GPU(s) available.' % torch.cuda.device_count())
	    print('We will use the GPU:', torch.cuda.get_device_name(0))
	else:
	    print('No GPU available, using the CPU instead.')
	    device = torch.device("cpu")
	
	train=pd.read_json('/content/drive/My Drive/DL/snli_1.0/snli_1.0_train.jsonl',encoding='utf-8',lines=True)
	valid=pd.read_json('/content/drive/My Drive/DL/snli_1.0/snli_1.0_dev.jsonl',encoding='utf-8',lines=True)
	test=pd.read_json('/content/drive/My Drive/DL/snli_1.0/snli_1.0_test.jsonl',encoding='utf-8',lines=True)
	target={'entailment': 0,  'neutral': 1, 'contradiction': 2}
	
	train['annotator_labels']=train['annotator_labels'].apply(lambda text: " ".join(text))
	valid['annotator_labels']=valid['annotator_labels'].apply(lambda text: " ".join(text))
	test['annotator_labels']=test['annotator_labels'].apply(lambda text: " ".join(text))
	train.drop_duplicates()
	valid.drop_duplicates()
	test.drop_duplicates()
	train.query('gold_label!="-"',inplace= True)
	valid.query('gold_label!="-"',inplace= True)
	test.query('gold_label!="-"',inplace= True)
	train['sentence']=train['sentence1'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))+" "+train['sentence2'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))
	valid['sentence']=valid['sentence1'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))+" "+valid['sentence2'].apply(lambda x:x.lower().translate(str.maketrans('','',string.punctuation)))
	test['sentence']=test['sentence1'].apply(lambda x:  x.lower().translate(str.maketrans('','',string.punctuation)))+" "+test['sentence2'].apply(lambda x:  x.lower().translate(str.maketrans('','',string.punctuation)))
	train=train.drop(columns=['annotator_labels','captionID','pairID','sentence1_binary_parse','sentence1_parse','sentence2_binary_parse','sentence2_parse'])
	valid=valid.drop(columns=['annotator_labels','captionID','pairID','sentence1_binary_parse','sentence1_parse','sentence2_binary_parse','sentence2_parse'])
	test=test.drop(columns=['annotator_labels','captionID','pairID','sentence1_binary_parse','sentence1_parse','sentence2_binary_parse','sentence2_parse'])
	train['gold_label']=train['gold_label'].apply(lambda x:target[x])
	test['gold_label']=test['gold_label'].apply(lambda x:target[x])
	valid['gold_label']=valid['gold_label'].apply(lambda x: target[x])
	
	sentences = train['sentence'].values
	labels = train['gold_label'].values
	test_x=test['sentence'].values
	test_y=test['gold_label'].values
	
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	
	input_ids = []
	attention_masks = []
	
	for sent in sentences:
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
	labels = torch.tensor(labels)
	
	dataset = TensorDataset(input_ids, attention_masks, labels)
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	batch_size = 8
	train_dataloader = DataLoader(
	            train_dataset,  
	            sampler = RandomSampler(train_dataset), 
	            batch_size = batch_size 
	        )
	validation_dataloader = DataLoader(
	            val_dataset, 
	            sampler = SequentialSampler(val_dataset), 
	            batch_size = batch_size 
	        )
	
	
	model = BertForSequenceClassification.from_pretrained(
	    "bert-base-uncased",
	    num_labels = 3, 
	    output_attentions = False, 
	    output_hidden_states = False, 
	)
	
	model.to(device)
	optimizer = AdamW(model.parameters(),
	                  lr = 2e-5, 
	                  eps = 1e-8 
	                )
	
	epochs = 1
	total_steps = len(train_dataloader) * epochs
	seed_val = 42
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	training_stats = []
	
	
	for epoch_i in range(0, epochs):
	    
	
	    print("")
	    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
	    print('Training...')
	    total_train_loss = 0
	    model.train()
	    for step, batch in enumerate(train_dataloader):
	        if step % 40 == 0 and not step == 0:
	            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))
	        
	        b_input_ids = batch[0].to(device)
	        b_input_mask = batch[1].to(device)
	        b_labels = batch[2].to(device)
	        model.zero_grad()        
	        loss, logits = model(b_input_ids, 
	                             token_type_ids=None, 
	                             attention_mask=b_input_mask, 
	                             labels=b_labels)
	        total_train_loss += loss.item()
	        loss.backward()
	        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
	        optimizer.step()
	        scheduler.step()
	    avg_train_loss = total_train_loss / len(train_dataloader)            
	    training_time = format_time(time.time() - t0)
	    print("")
	    print("  Average training loss: {0:.2f}".format(avg_train_loss))
	    print("  Training epcoh took: {:}".format(training_time))
	    print("")
	    print("Running Validation...")
	    model.eval()
	    total_eval_accuracy = 0
	    total_eval_loss = 0
	    nb_eval_steps = 0
	    for batch in validation_dataloader:
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
	        
	
	    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
	    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
	    avg_val_loss = total_eval_loss / len(validation_dataloader)
	    validation_time = format_time(time.time() - t0)
	    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
	
	    training_stats.append(
	        {
	            'epoch': epoch_i + 1,
	            'Training Loss': avg_train_loss,
	            'Valid. Loss': avg_val_loss,
	            'Valid. Accur.': avg_val_accuracy,
	        }
	    )
	
	print("")
	print("Training complete!")
	
	
	model_to_save = model.module if hasattr(model, 'module') else model  
	model_to_save.save_pretrained('/content/drive/My Drive/DL')
	tokenizer.save_pretrained('/content/drive/My Drive/DL')
	
	# model = BertForSequenceClassification.from_pretrained('/content/drive/My Drive/DL')
	# tokenizer =  BertTokenizer.from_pretrained('/content/drive/My Drive/DL')
	# model.to(device)
	
	