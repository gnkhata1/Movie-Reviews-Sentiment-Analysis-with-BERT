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


# next three functions are loading imdb dataset
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)
  
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))

    return train_df, test_df


trains, tests = download_and_load_datasets()

whole_data = np.concatenate((trains, tests), axis=0)


def binary_tree_split(whole_data):
    rows_count = whole_data.shape[0]

    for i in range (rows_count):
        if whole_data[i][1] == '1' or whole_data[i][1] == '2':
            whole_data[i][2] = "Highly Negative"
        elif whole_data[i][1] == '3' or whole_data[i][1] == '4' or whole_data[i][1] == '5':
            whole_data[i][2] = 'Negative'
        elif whole_data[i][1] == '6' or whole_data[i][1] == '7' or whole_data[i][1] == '8':
            whole_data[i][2] = 'Positive'
        else:
            whole_data[i][2] = 'Highly Positive'     
    return whole_data

whole_data = binary_tree_split(whole_data)

df =  pd.DataFrame(whole_data, columns = ['news','sentiment','category'])

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
    mapLabels.drop(['news'], axis = 1, inplace = True)
    label2Index = mapLabels.to_dict(orient='index')

    index2label = {}

    for key in label2Index:
        index2label[key[1]] = key[0]

    label2Index = {v: k for k, v in index2label.items()}

    df.rename(columns = {'label' : 'labels', 'news' : 'text'}, inplace = True)
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
SEQ_LEN = 256

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


#creating dataset that will be fed into bert
dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
dataset = dataset.map(map_func)


def data_split(dataset):
    DS_LEN = len(list(dataset))
    
    SPLIT = .85
    test = dataset.skip(round(DS_LEN * SPLIT)).shuffle(100).batch(64)
    trainevalu = dataset.take(round(DS_LEN * SPLIT))

    DS_LEN2 = len(list(trainevalu))

    train = trainevalu.take(round(DS_LEN2 * SPLIT)).shuffle(100).batch(64).repeat(2)
    evalu = trainevalu.skip(round(DS_LEN2 * SPLIT)).shuffle(100).batch(64)
    
    return test, train, evalu, DS_LEN

test, train, evalu, DS_LEN = data_split(dataset)


def build_model():
    tranformersPreTrainedModelName = 'bert-base-uncased'
    bertConfig = BertConfig.from_pretrained(tranformersPreTrainedModelName
                                            , output_hidden_states=True
                                            , num_lables=4
                                            , max_length=SEQ_LEN
                                            , label2id=label2Index
                                            , id2label=index2label
                                            )

    
    bert = TFBertForSequenceClassification.from_pretrained(tranformersPreTrainedModelName, config=bertConfig)
    return bert


bert = build_model()

def add_inputLayers_to_model(SEQ_LEN, bert):
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

#training the model
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
    print("Training for 4 classes on IMDB-4 with BERT-BiLSTM: ", timedelta(seconds=end-start))
    
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

# Generate predictions on test data
predictions = model.predict(test)
test_len = len(list(test))
count_batch  = 0     
y_pred = []#store predicted label
y_test = []#store test label

batch_size = 32
check_batch = 0
out_of_bounds = len(predictions)%batch_size 
index = 0

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

#finding overall polarity for four-class classification
def four_clas_overal_polarity(predictions):
    rows_count = predictions.shape[0]
    
    high_neg_count = 0 #count for high negative polarity
    neg_count = 0 #count for negative polarity
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
        
        else:
            pos_count =  pos_count + 1
    
    print("High negative polarity has: ", high_neg_count)
    print("Negative polarity has: ", neg_count)
    print("Positive polarity has: ", pos_count)
    print("High positive polarity has: ", high_pos_count)
    
    #Finding ovrerall polarity using comparisons 
    total_sents_count = high_neg_count + high_pos_count + neg_count + pos_count


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
four_clas_overal_polarity(label)

