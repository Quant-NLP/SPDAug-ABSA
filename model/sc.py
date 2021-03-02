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

pos_token = 'positive'
neg_token = 'negative'
neu_token = 'neutral'
conf_token = 'conflict'

model_type = 'bert-base-uncased'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ABSA_task = sys.argv[1]
dataset = sys.argv[2]
DPM_type = sys.argv[3]
replacement_strategy = sys.argv[4]

input_path = './dataset/' + ABSA_task + '/' + dataset + '/'

MAX_LEN = 50 
batch_size = 100
std_strength = 0
seed_val = 2020

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

def transform_sentiment_y_true(df):
    df['y_true'] = 0
    for i in range(len(df)):
        if df['sentiment'].iloc[i] == pos_token:
            df['y_true'].iloc[i] = 0
        elif df['sentiment'].iloc[i] == neg_token:
            df['y_true'].iloc[i] = 1
        elif df['sentiment'].iloc[i] == neu_token:
            df['y_true'].iloc[i] = 2
        elif df['sentiment'].iloc[i] == conf_token:
            df['y_true'].iloc[i] = 3
    return df

def transform_sentiment_y_pred(df):
    df['sentiment_pred'] = 0
    for i in range(len(df)):
        if df['y_pred'].iloc[i] == 0:
            df['sentiment_pred'].iloc[i] = pos_token
        elif df['y_pred'].iloc[i] == 1:
            df['sentiment_pred'].iloc[i] = neg_token
        elif df['y_pred'].iloc[i] == 2:
            df['sentiment_pred'].iloc[i] = neu_token
        elif df['y_pred'].iloc[i] == 3:
            df['sentiment_pred'].iloc[i] = conf_token
    return df



train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 0
                }

class Data(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.y_true
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def training(epoch, model, training_loader, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)


        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    #print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

def testing(model, testing_loader, test, loss_function):

    prediction = []
    true = []

    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            #print(outputs.shape)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            prediction.extend(big_idx.cpu().numpy().tolist())
            true.extend(targets.cpu().numpy().tolist())

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Testing Loss Epoch: {epoch_loss}")
    print(f"Testing Accuracy Epoch: {epoch_accu}")

    test['y_pred'] = prediction
    test = transform_sentiment_y_pred(test)

    #print(classification_report(test['sentiment'], test['sentiment_pred']))

    return test



def AE(df):
    
    model_type = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_type)
    model = BertModel.from_pretrained(model_type, return_dict=True)
    mask_model = BertForMaskedLM.from_pretrained(model_type, return_dict=True)

    optimizer = AdamW(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mask_model.to(device)
    
    df['mask_text'] = 0
    df['auxiliary_text'] = 0

    for i in range(len(df)):
        sentiment = df['sentiment'].iloc[i]
        
        aspect = 'sentence'

        if DPM_type == 'Senti':
            mask_sent = 'the polarity of the ' + aspect + ' is ' + '[MASK]' + ' [SEP] '
            auxiliary_sent = 'the polarity of the ' + aspect + ' is ' + sentiment + ' [SEP] '

        df['mask_text'].iloc[i] = mask_sent + df['text'].iloc[i]
        df['auxiliary_text'].iloc[i] = auxiliary_sent + df['text'].iloc[i]


    df['distance'] = 0
    df = df.astype('object')

    for i in range(len(df)): 

        sep_id = tokenizer('[SEP]')['input_ids'][1]

        mask_id = tokenizer('[MASK]')['input_ids'][1]

        tokenized = df['mask_text'][i:i+1].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=MAX_LEN, truncation=True)))

        sep_index = tokenized[i].index(sep_id)
        mask_index = tokenized[i].index(mask_id)

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
            if df['distance'].iloc[i][dis_index][1] < mask_threshold:
                perturbed_mask_index.append(dis_index)

        df['perturbed_mask_index'].iloc[i] = perturbed_mask_index
        
    df['augment_token_id'] = 0
    df = df.astype('object')

    for i in range(len(df)):
        tokenized = tokenizer.encode(df['auxiliary_text'].iloc[i])
        tokenized = torch.Tensor(tokenized).unsqueeze(0).to(torch.int64).to(device)
        augment_tokenized = tokenizer.encode(df['auxiliary_text'].iloc[i])

        for j in range(len(df['perturbed_mask_index'].iloc[i])):
            mask_tokenized = tokenizer.encode(df['auxiliary_text'].iloc[i])
            sep_index = mask_tokenized.index(sep_id)
            perturbed_mask_index = df['perturbed_mask_index'].iloc[i][j] + sep_index + 1
            mask_tokenized[perturbed_mask_index] = mask_id

            mask_tokenized = torch.Tensor(mask_tokenized).unsqueeze(0).to(torch.int64).to(device)

            outputs = mask_model(mask_tokenized, labels = tokenized)
            augment_tokenized[perturbed_mask_index] = int(outputs.logits[:, perturbed_mask_index, :].argmax().cpu().numpy())

        df['augment_token_id'].iloc[i] = augment_tokenized

    df['augment_text'] = 0
    df = df.astype('object')

    for i in range(len(df)):
        sep_index = df['augment_token_id'].iloc[i].index(sep_id)
        df['augment_text'].iloc[i] = tokenizer.decode(df['augment_token_id'].iloc[i][sep_index+1:-1])
    return df



def Seq2Seq(df):
    model_type = 'facebook/bart-large'

    tokenizer = BartTokenizer.from_pretrained(model_type)
    model = BartModel.from_pretrained(model_type)
    mask_model = BartForConditionalGeneration.from_pretrained(model_type)

    sep_token = '</s>'
    mask_token = '<mask>'

    mask_id = tokenizer(mask_token, return_tensors='pt')['input_ids'][0][1]
    sep_id = tokenizer(sep_token, return_tensors='pt')['input_ids'][0][1]

    optimizer = AdamW(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mask_model.to(device)
    
    df['mask_text'] = 0
    df['auxiliary_text'] = 0

    for i in range(len(df)):
        sentiment = df['sentiment'].iloc[i]
        
        aspect = 'sentence'

        if DPM_type == 'Senti':
            mask_sent = 'the polarity of the ' + aspect + ' is ' + mask_token + ' ' + sep_token  + ' '
            auxiliary_sent = 'the polarity of the ' + aspect + ' is ' + sentiment + ' ' + sep_token  + ' '
   
        df['mask_text'].iloc[i] = mask_sent + df['text'].iloc[i]
        df['auxiliary_text'].iloc[i] = auxiliary_sent + df['text'].iloc[i]
        
    df['distance'] = 0
    df = df.astype('object')

    for i in range(len(df)): 

        tokenized = df['mask_text'][i:i+1].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=MAX_LEN, truncation=True)))

        sep_index = tokenized[i].index(sep_id)
        mask_index = tokenized[i].index(mask_id)

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
            if df['distance'].iloc[i][dis_index][1] < mask_threshold:
                perturbed_mask_index.append(dis_index)

        df['perturbed_mask_index'].iloc[i] = perturbed_mask_index
    
    
    df['augment_token_id'] = 0
    df = df.astype('object')

    for i in range(len(df)):
        tokenized = tokenizer.encode(df['auxiliary_text'].iloc[i])
        tokenized = torch.Tensor(tokenized).unsqueeze(0).to(torch.int64).to(device)
        augment_tokenized = tokenizer.encode(df['auxiliary_text'].iloc[i])

        mask_tokenized = tokenizer.encode(df['auxiliary_text'].iloc[i])
        sep_index = mask_tokenized.index(sep_id)

        for j in range(len(df['perturbed_mask_index'].iloc[i])):
            perturbed_mask_index = df['perturbed_mask_index'].iloc[i][j] + sep_index + 1
            mask_tokenized[perturbed_mask_index] = mask_id

        mask_tokenized = torch.Tensor(mask_tokenized).unsqueeze(0).to(torch.int64).to(device)
        logits = mask_model(mask_tokenized).logits

        for j in range(len(df['perturbed_mask_index'].iloc[i])):
            perturbed_mask_index = df['perturbed_mask_index'].iloc[i][j] + sep_index + 1
            probs = logits[0, perturbed_mask_index].softmax(dim=0)
            values, predictions = probs.topk(1)
            augment_tokenized[perturbed_mask_index] = int(predictions.cpu().numpy())

        df['augment_token_id'].iloc[i] = augment_tokenized
    
    
    df['augment_text'] = 0
    df = df.astype('object')

    for i in range(len(df)):
        sep_index = df['augment_token_id'].iloc[i].index(sep_id)
        df['augment_text'].iloc[i] = tokenizer.decode(df['augment_token_id'].iloc[i][sep_index+1:-1])

    return df


train = pd.read_csv(input_path + 'train.csv')
test = pd.read_csv(input_path + 'test.csv')
way = len(train['sentiment'].value_counts())

print('Run DPM  ... ')
if replacement_strategy == 'AE':
    train = AE(train)
elif replacement_strategy == 'Seq2Seq':
    train = Seq2Seq(train)

#train.to_csv('./train' + ABSA_task + dataset + DPM_type + replacement_strategy + '.csv', index = 0)
#train = pd.read_csv('./train' + ABSA_task + dataset + DPM_type + replacement_strategy + '.csv')

train = transform_sentiment_y_true(train)
test = transform_sentiment_y_true(test)

train_data = train[['text', 'y_true']]
test_data = test[['text', 'y_true']]

aug_train_data = train[['augment_text', 'y_true']]
aug_train_data.columns = ['text', 'y_true']

train_data = pd.concat([train_data, aug_train_data], axis=0).reset_index(drop=True) 

tokenizer = BertTokenizer.from_pretrained(model_type, truncation=True, do_lower_case=True)

training_set = Data(train_data, tokenizer, MAX_LEN)
testing_set = Data(test_data, tokenizer, MAX_LEN)

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

feature_dim = 768
class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        #self.l1 = RobertaModel.from_pretrained(model_type)
        self.l1 = BertModel.from_pretrained(model_type)
        self.pre_classifier = torch.nn.Linear(feature_dim, feature_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(feature_dim, way)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = BertClass()
model.to(device)

LEARNING_RATE = 2e-05

loss_function = CrossEntropyLoss()
optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)

print('Run Model ... ')
EPOCHS = 3
for epoch in range(EPOCHS):
    training(epoch, model, training_loader, loss_function)

test = testing(model, testing_loader, test, loss_function)
