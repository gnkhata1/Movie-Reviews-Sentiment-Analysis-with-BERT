#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import TFBertForSequenceClassification
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import datasets
from datasets import load_dataset

dataset = load_dataset("sst", "default")

#preprocessing the sst dataset 
#To use amazon-5 check Utilities.py
def load_data(dataset):
    train_d = dataset["train"]
    test_d = dataset["test"]
    val_d = dataset["validation"]

    #obtaining columns of the dataset
    sentence_tr, label_tr = train_d["sentence"], train_d["label"]
    sentence_ts, label_ts = test_d["sentence"], test_d["label"]
    sentence_vl, label_vl = val_d["sentence"], val_d["label"]

    #changing to numpy array
    sentence_tr, label_tr = np.asarray(sentence_tr), np.asarray(label_tr)
    sentence_ts, label_ts = np.asarray(sentence_ts), np.asarray(label_ts)
    sentence_vl, label_vl = np.asarray(sentence_vl), np.asarray(label_vl)

    #reshaping arrays to 2D
    sentence_tr, label_tr = sentence_tr.reshape(-1, 1), label_tr.reshape(-1, 1)
    sentence_ts, label_ts = sentence_ts.reshape(-1, 1), label_ts.reshape(-1, 1)
    sentence_vl, label_vl = sentence_vl.reshape(-1, 1), label_vl.reshape(-1, 1)

    #concatenating reviews and labels
    train = np.concatenate((sentence_tr, label_tr), axis=1)
    test = np.concatenate((sentence_ts, label_ts), axis=1)
    val = np.concatenate((sentence_vl, label_vl), axis=1)

    #shuffling data    
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)

    whole_data = np.concatenate((train, test, val))

    return whole_data


whole_data = load_data(dataset)

#creating 5 class sst:
def create_sst_5(whole_data):
    rows_count = whole_data.shape[0]
    count_05 =0
    for i in range (rows_count):
        if float(whole_data[i][1]) <= 0.2:
            whole_data[i][1] = "H Neg"

        elif float(whole_data[i][1]) <= 0.4:
            whole_data[i][1] = 'Neg'

        elif float(whole_data[i][1]) <= 0.6:
            whole_data[i][1] = 'Neut'

        elif float(whole_data[i][1]) <= 0.8:
            whole_data[i][1] = 'Pos'

        elif float(whole_data[i][1]) <= 1.0:
            whole_data[i][1] = 'H Pos'

    return whole_data

whole_data = create_sst_5(whole_data)

df =  pd.DataFrame(whole_data, columns = ['review','category'])

#viewing class distribution
y = df['category']

count = y.value_counts()
count.plot.bar()
plt.ylabel('Number of records')
plt.xlabel('Target Class')
plt.show()

def map_to_categorical(df):
    df['label'] = pd.Categorical(df.category, ordered=True).codes
    df['label'].unique()

    mapLabels = pd.DataFrame(df.groupby(['category', 'label']).count())
    mapLabels.drop(['review'], axis = 1, inplace = True)
    label2Index = mapLabels.to_dict(orient='index')

    index2label = {}
    for key in label2Index:
        index2label[key[1]] = key[0]

    label2Index = {v: k for k, v in index2label.items()}
    df.rename(columns = {'label' : 'labels', 'review' : 'text'}, inplace = True)

    df = df[['text','labels']]
    return df, label2Index, index2label

df, label2Index, index2label =  map_to_categorical(df)

def one_hot_encoding_labels(df):
    arr = df['labels'].values
    labels = np.zeros((arr.size, arr.max() + 1), dtype=int)
    labels[np.arange(arr.size), arr] = 1
    return labels, arr

labels, arr = one_hot_encoding_labels(df)

seqlen = df['text'].apply(lambda x: len(x.split()))
SEQ_LEN = 128

def tokenize_data(SEQ_LEN, df):
    tranformersPreTrainedModelName = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(tranformersPreTrainedModelName) 
    Xids = np.zeros((len(df), SEQ_LEN), dtype=int)
    Xmask = np.zeros((len(df), SEQ_LEN), dtype=int)
    for i, sequence in enumerate(df['text']):
      tokens = tokenizer.encode_plus(sequence
                                   ,max_length = SEQ_LEN          
                                   ,truncation=True               
                                   ,padding='max_length'          
                                   ,add_special_tokens=True       
                                   ,return_token_type_ids = False 
                                   ,return_attention_mask = True
                                   ,return_tensors='tf')

      Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']
    
    return Xids, Xmask, tokenizer

Xids, Xmask, tokenizer = tokenize_data(SEQ_LEN, df)


def map_func(input_ids, masks, labels):
  return {'input_ids': input_ids, 'attention_mask': masks}, labels

#creating dataset to be fed into bert
dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
dataset = dataset.map(map_func)

#data split
def data_split(dataset):
    DS_LEN = len(list(dataset))
    
    SPLIT = .85

    # take or skip the specified number of batches to split by factor
    test = dataset.skip(round(DS_LEN * SPLIT)).shuffle(100).batch(64)
    trainevalu = dataset.take(round(DS_LEN * SPLIT))

    DS_LEN2 = len(list(trainevalu))

    train = trainevalu.take(round(DS_LEN2 * SPLIT)).shuffle(100).batch(64).repeat(2)
    evalu = trainevalu.skip(round(DS_LEN2 * SPLIT)).shuffle(100).batch(64)
    
    return test, train, evalu, DS_LEN

    #del dataset

test, train, evalu, DS_LEN = data_split(dataset)

#build model
def build_model():
    tranformersPreTrainedModelName = 'bert-base-uncased'
    bertConfig = BertConfig.from_pretrained(tranformersPreTrainedModelName
                                            , output_hidden_states=True
                                            , num_lables=5
                                            , max_length=SEQ_LEN
                                            , label2id=label2Index
                                            , id2label=index2label
                                            )

    
    bert = TFBertForSequenceClassification.from_pretrained(tranformersPreTrainedModelName, config=bertConfig)
    return bert

bert = build_model()

def add_inputLayers_to_model(SEQ_LEN, bert):
    # build 2 input layers to Bert Model where name needs to match the input values in the dataset
    input_ids = tf.keras.Input(shape=(SEQ_LEN,), name = 'input_ids', dtype='int32')
    mask = tf.keras.Input(shape=(SEQ_LEN,), name = 'attention_mask', dtype='int32')

    embedings = bert.layers[0](input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.Dropout(0.5)(embedings)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768))(X)
    y = tf.keras.layers.Dense(len(label2Index), activation='softmax', name='outputs')(X)

    model = tf.keras.Model(inputs=[input_ids,mask], outputs=y)
    
    return model

model = add_inputLayers_to_model(SEQ_LEN, bert)
model.layers[2].trainable = False

#training our model
def train_model(model, train, evalu):
    loss=tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
        )

    start = timer()

    history = model.fit(train
                        , validation_data=evalu
                        , epochs=4)

    end = timer()
    print("Training for 5 classes on sst-5: ", timedelta(seconds=end-start))
    
    return model, history

model, history = train_model(model, train, evalu)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

results = model.evaluate(test, batch_size=64)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
predictions = model.predict(test)
print("predictions shape:", predictions.shape)

#predicting on new data
test_len = len(list(test))
count_batch  = 0     
y_pred = []#store predicted label
y_test = []#store test label
#round(test_len * 1.0)
batch_size = 64
check_batch = 0
out_of_bounds = len(predictions)%batch_size 
index = 0
#print(out_of_bounds)

for item in test.take(test_len):
    
    check_batch = check_batch+1
    
    for i in range(batch_size):
        if check_batch == test_len and i==out_of_bounds:
            break
        
        actualLabelIdx=np.argmax(item[1][i])
        
        index = batch_size*count_batch+i #index of a prediction
        
        predicLabelIdx=np.argmax(predictions[index])
        
        y_pred.append(predicLabelIdx)
        y_test.append(actualLabelIdx)
    
    count_batch = count_batch +1

label = tf.argmax(predictions, axis=1)
eval_report(label, y_test)

#finding overall polarity for five-class classification
def five_clas_overal_polarity(predictions):
    rows_count = predictions.shape[0]
    
    high_neg_count = 0 #count for high negative polarity
    neg_count = 0 #count for negative polarity
    neut_count = 0 #count for neutral polarity
    pos_count = 0 #count for positive polarity
    high_pos_count = 0 #count for high positive polarity
    
    #counting occurrence of every polarity in predictions
    for i in range (rows_count):
        if predictions[i] == 0:
            high_neg_count = high_neg_count +1

        elif predictions[i] == 1:
            high_pos_count = high_pos_count + 1
            
        elif predictions[i] == 2:
            neg_count =  neg_count + 1
            
        elif predictions[i] == 3:
            neut_count =  neut_count + 1
        
        elif predictions[i] == 3:
            pos_count =  pos_count + 1
            
    print("High negative polarity has: ", high_neg_count)
    print("Negative polarity has: ", neg_count)
    print("Neutral polarity has: ", neut_count)
    print("Positive polarity has: ", pos_count)
    print("High positive polarity has: ", high_pos_count)
    
    #Finding ovrerall polarity using comparisons 
    total_sents_count = high_neg_count + high_pos_count+neut_count+ neg_count + pos_count
    if(neut_count/total_sents_count*100)>0.85:
        polarity = "Neutral"
        
    else:
        if(high_neg_count+neg_count)>1.5*(high_pos_count+pos_count):
            if high_neg_count>1.5*neg_count:
                polarity = "Highly negative"

            else:
                polarity = "Negative"

        elif(high_pos_count+pos_count)>1.5*(high_neg_count+neg_count):
            if high_pos_count>1.5*pos_count:
                polarity = "Highly positive"

            else:
                polarity = "Positive"

        else:
            polarity = "Neutral"

    print("Overall polarity is ", polarity)

label = tf.argmax(predictions, axis=1)
five_clas_overal_polarity(label)

