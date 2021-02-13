import numpy as np
import pandas as pd
import sys
import os
import json
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d,BatchNorm1d,MaxPool1d,ReLU,Dropout
from torch.optim import Adam
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertModel
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import  BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BartTokenizer, BartForConditionalGeneration, BartModel
from torch.utils.data import Dataset
import random
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from seqeval.metrics import classification_report as token_classification_report
from seqeval.metrics import accuracy_score, f1_score, recall_score, precision_score
sys.path.append("../") 
from bert_sklearn import BertTokenClassifier, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten(l):
    return [item for sublist in l for item in sublist]

ABSA_task = sys.argv[1]
dataset = sys.argv[2]
DPM_type = sys.argv[3]
replacement_strategy = sys.argv[4]

input_path = './dataset/' + ABSA_task + '/' + dataset + '/'

print('ABSA_task : ', ABSA_task,
      '\ndataset : ', dataset,
      '\nDPM_type : ', DPM_type,
      '\nreplacement_strategy : ', replacement_strategy)

batch_size = 85
MAX_LEN = 50
std_strength = 0
seed_val = 2017

def read_json(data_path):

    with open(data_path + 'train'  + '.json') as reader:
        jf = json.loads(reader.read())
    train =  pd.DataFrame(jf).reset_index(drop=True)

    with open(data_path + 'test' + '.json') as reader:
        jf = json.loads(reader.read())
    test =  pd.DataFrame(jf).reset_index(drop=True)
    
    return train.reset_index(drop=True), test.reset_index(drop=True)

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def dist(x, y):
    # inner product or cos similarity
    return np.sqrt(((x - y)**2).sum())

def mask_embedding(model, padded, mask_index):
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded).to(device)  
    attention_mask = torch.tensor(attention_mask).to(device)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        
    return last_hidden_states[0][:, mask_index, :].cpu().numpy()

def calculate_threshold(distance, std_strength):
    return distance.mean() + distance.std() * std_strength

def Seq2Seq(df):
    model_type = 'bart-large'

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartModel.from_pretrained('facebook/bart-large')
    mask_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    sep_token = '</s>'
    mask_token = '<mask>'

    mask_id = tokenizer(mask_token, return_tensors='pt')['input_ids'][0][1]
    sep_id = tokenizer(sep_token, return_tensors='pt')['input_ids'][0][1]

    optimizer = AdamW(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mask_model.to(device)
    
    
    auxiliary_tokens = ['the', 'aspect', 'term', 'is'] 

    df['mask_tokens'] = 0
    df['auxiliary_tokens'] = 0
    df = df.astype('object')

    for i in range(len(df)):

    #for j in range(len(df['aspect_terms'].iloc[i])):
        auxiliary_sents = []
        for j in range(len(df['aspect_terms'].iloc[i])): 
            aspect_terms = df['aspect_terms'].iloc[i][j]
            auxiliary_sent = auxiliary_tokens + [aspect_terms] + [sep_token] + df['tokens'].iloc[i]
            auxiliary_sents.append(auxiliary_sent)

        mask_sent = auxiliary_tokens + [mask_token] + [sep_token] + df['tokens'].iloc[i]
        df['mask_tokens'].iloc[i] = mask_sent 
        df['auxiliary_tokens'].iloc[i] = auxiliary_sents
    
    df['distance'] = 0
    df = df.astype('object')

    for i in range(len(df)): 

        tokenized = tokenizer.encode(df['mask_tokens'].iloc[i])

        sep_index = tokenized.index(sep_id)
        mask_index = tokenized.index(mask_id)

        tokenized = pd.Series([tokenized])

        padded = pad_sequences(tokenized, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded).to(device)  
        attention_mask = torch.tensor(attention_mask).to(device)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        original_mask_embedding = last_hidden_states[0][:, mask_index, :].cpu().numpy()


        distance = []

        for pertubed_index in range(sep_index+1, MAX_LEN):
            padded = pad_sequences(tokenized, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
            if padded[0][pertubed_index] != 0 and padded[0][pertubed_index] != sep_id:
                #print(padded.shape)
                cur_id = padded[0][pertubed_index]
                padded[0][pertubed_index] = mask_id

                cur_embedding = mask_embedding(model, padded, mask_index)
                d = dist(original_mask_embedding, cur_embedding)
                distance.append((cur_id, d))

        df['distance'].iloc[i] = distance
    
    df['perturbed_mask_index'] = 0
    df = df.astype('object')

    for i in range(len(df)):
        perturbed_mask_index = []
        mask_threshold = calculate_threshold(np.array(df['distance'].iloc[i])[:, 1], std_strength)
        for dis_index in range(len(df['distance'].iloc[i])):
            if df['distance'].iloc[i][dis_index][1] < mask_threshold and df['labels'].iloc[i][dis_index] != 'B' and df['labels'].iloc[i][dis_index] != 'I':
                perturbed_mask_index.append(dis_index)

        df['perturbed_mask_index'].iloc[i] = perturbed_mask_index
        
    df['augment_token_id'] = 0
    df = df.astype('object')

    for i in range(len(df)):

        augment_tokenizeds = []

        for j in range(len(df['aspect_terms'].iloc[i])):

            tokenized = tokenizer.encode(df['auxiliary_tokens'].iloc[i][j])
            tokenized = torch.Tensor(tokenized).unsqueeze(0).to(torch.int64).to(device)
            augment_tokenized = tokenizer.encode(df['auxiliary_tokens'].iloc[i][j])

            for k in range(len(df['perturbed_mask_index'].iloc[i])):
                mask_tokenized = tokenizer.encode(df['auxiliary_tokens'].iloc[i][j])
                sep_index = mask_tokenized.index(sep_id)
                perturbed_mask_index = df['perturbed_mask_index'].iloc[i][k] + sep_index + 1
                mask_tokenized[perturbed_mask_index] = mask_id

                mask_tokenized = torch.Tensor(mask_tokenized).unsqueeze(0).to(torch.int64).to(device)

                logits = mask_model(mask_tokenized).logits

                probs = logits[0, perturbed_mask_index].softmax(dim=0)
                values, predictions = probs.topk(1)
                augment_tokenized[perturbed_mask_index] = int(predictions.cpu().numpy())

            augment_tokenizeds.append(augment_tokenized)

        df['augment_token_id'].iloc[i] = augment_tokenizeds
    
    df['augment_tokens'] = 0
    df = df.astype('object')

    for i in range(len(df)):

        tokens_lists = []

        for j in range(len(df['aspect_terms'].iloc[i])):

            tokens_list = []

            for k in range(1, len(df['augment_token_id'].iloc[i][j]) - 1):
                tokens_list.append(tokenizer.decode([df['augment_token_id'].iloc[i][j][k]]))

            sep_index = tokens_list.index(sep_token)
            tokens_list = tokens_list[sep_index + 1 : ]
            tokens_lists.append(tokens_list)

        df['augment_tokens'].iloc[i] = tokens_lists
        
    return df
    
def AE(df):
    
    model_type = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_type)
    model = BertModel.from_pretrained(model_type, return_dict=True)
    mask_model = BertForMaskedLM.from_pretrained(model_type, return_dict=True)
    
    sep_token = '[SEP]'
    mask_token = '[MASK]'

    mask_id = tokenizer(mask_token)['input_ids'][1]
    sep_id = tokenizer(sep_token)['input_ids'][1]

    optimizer = AdamW(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mask_model.to(device)

    auxiliary_tokens = ['the', 'aspect', 'term', 'is'] 

    df['mask_tokens'] = 0
    df['auxiliary_tokens'] = 0
    df = df.astype('object')

    for i in range(len(df)):

    #for j in range(len(df['aspect_terms'].iloc[i])):
        auxiliary_sents = []
        for j in range(len(df['aspect_terms'].iloc[i])): 
            aspect_terms = df['aspect_terms'].iloc[i][j]
            auxiliary_sent = auxiliary_tokens + [aspect_terms] + [sep_token] + df['tokens'].iloc[i]
            auxiliary_sents.append(auxiliary_sent)

        mask_sent = auxiliary_tokens + [mask_token] + [sep_token] + df['tokens'].iloc[i]
        df['mask_tokens'].iloc[i] = mask_sent 
        df['auxiliary_tokens'].iloc[i] = auxiliary_sents


    df['distance'] = 0
    df = df.astype('object')

    for i in range(len(df)): 

        tokenized = tokenizer.encode(df['mask_tokens'].iloc[i])

        sep_index = tokenized.index(sep_id)
        mask_index = tokenized.index(mask_id)

        tokenized = pd.Series([tokenized])

        padded = pad_sequences(tokenized, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded).to(device)  
        attention_mask = torch.tensor(attention_mask).to(device)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        original_mask_embedding = last_hidden_states[0][:, mask_index, :].cpu().numpy()


        distance = []

        for pertubed_index in range(sep_index+1, MAX_LEN):
            padded = pad_sequences(tokenized, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
            if padded[0][pertubed_index] != 0 and padded[0][pertubed_index] != sep_id:
                #print(padded.shape)
                cur_id = padded[0][pertubed_index]
                padded[0][pertubed_index] = mask_id

                cur_embedding = mask_embedding(model, padded, mask_index)
                d = dist(original_mask_embedding, cur_embedding)
                distance.append((cur_id, d))

        df['distance'].iloc[i] = distance

    df['perturbed_mask_index'] = 0
    df = df.astype('object')

    for i in range(len(df)):
        perturbed_mask_index = []
        mask_threshold = calculate_threshold(np.array(df['distance'].iloc[i])[:, 1], std_strength)
        for dis_index in range(len(df['distance'].iloc[i])):
            if df['distance'].iloc[i][dis_index][1] < mask_threshold and df['labels'].iloc[i][dis_index] != 'B' and df['labels'].iloc[i][dis_index] != 'I':
                perturbed_mask_index.append(dis_index)

        df['perturbed_mask_index'].iloc[i] = perturbed_mask_index

    df['augment_token_id'] = 0
    df = df.astype('object')

    for i in range(len(df)):

        augment_tokenizeds = []

        for j in range(len(df['aspect_terms'].iloc[i])):

            tokenized = tokenizer.encode(df['auxiliary_tokens'].iloc[i][j])
            tokenized = torch.Tensor(tokenized).unsqueeze(0).to(torch.int64).to(device)
            augment_tokenized = tokenizer.encode(df['auxiliary_tokens'].iloc[i][j])

            for k in range(len(df['perturbed_mask_index'].iloc[i])):
                mask_tokenized = tokenizer.encode(df['auxiliary_tokens'].iloc[i][j])
                sep_index = mask_tokenized.index(sep_id)
                perturbed_mask_index = df['perturbed_mask_index'].iloc[i][k] + sep_index + 1
                mask_tokenized[perturbed_mask_index] = mask_id

                mask_tokenized = torch.Tensor(mask_tokenized).unsqueeze(0).to(torch.int64).to(device)

                outputs = mask_model(mask_tokenized, labels = tokenized)
                augment_tokenized[perturbed_mask_index] = int(outputs.logits[:, perturbed_mask_index, :].argmax().cpu().numpy())

            augment_tokenizeds.append(augment_tokenized)

        df['augment_token_id'].iloc[i] = augment_tokenizeds

    df['augment_tokens'] = 0
    df = df.astype('object')

    for i in range(len(df)):

        tokens_lists = []

        for j in range(len(df['aspect_terms'].iloc[i])):

            tokens_list = []

            for k in range(1, len(df['augment_token_id'].iloc[i][j]) - 1):

                tokens_list.append(tokenizer.decode([df['augment_token_id'].iloc[i][j][k]]))

            sep_index = tokens_list.index(sep_token)
            tokens_list = tokens_list[sep_index + 1 : ]
            tokens_lists.append(tokens_list)

        df['augment_tokens'].iloc[i] = tokens_lists
        
        return df

    
train, test = read_json(input_path)
    
print('Run DPM  ... ')
if replacement_strategy == 'AE':
    train = AE(train)
elif replacement_strategy == 'Seq2Seq':
    train = Seq2Seq(train)
    
#train.to_json('./train' + ABSA_task + dataset + DPM_type + replacement_strategy + '.json')

aug_train = pd.DataFrame()

for i in range(len(train)):

    for j in range(len(train['aspect_terms'].iloc[i])):
        augment_tokens_labels = pd.DataFrame([[train['augment_tokens'].iloc[i][j]], [train['labels'].iloc[i]]])
        aug_train = pd.concat([aug_train, augment_tokens_labels], axis = 1)

aug_train = aug_train.T.reset_index(drop=True)
aug_train.columns = ['tokens', 'labels']

train = train[['tokens', 'labels']]
test = test[['tokens', 'labels']]

train = pd.concat([train, aug_train], axis=0).reset_index(drop=True)

X_train, y_train = train.tokens, train.labels
X_test, y_test = test.tokens, test.labels

label_list = np.unique(flatten(y_train))
label_list = list(label_list)
print("\nNER tags:",label_list)

# define model
model = BertTokenClassifier(bert_model='bert-base-uncased',
                    epochs=3,
                    learning_rate=5e-5,
                    train_batch_size=batch_size,
                    eval_batch_size=16,
                    validation_fraction=0.05,                            
                    label_list=label_list,
                    ignore_label=['O'],
                    random_state=seed_val)



model.max_seq_length = 50
model.gradient_accumulation_steps = 2
print(model)

# finetune model on train data
model.fit(X_train, y_train)

# score model on dev data
#f1_dev = model.score(X_dev, y_dev)
#print("Dev f1: %0.02f"%(f1_dev))

# score model on test data
f1_test = model.score(X_test, y_test)
print("Test f1: %0.02f"%(f1_test))

# get predictions on test data
y_preds = model.predict(X_test)

# calculate the probability of each class
y_probs = model.predict_proba(X_test)

# print report on classifier stats
print(token_classification_report(y_test, y_preds))

