#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import os
import re
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# Next three functions are loading original IMDd dataset
#To use other datasets check Utilities.py
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

train, test = download_and_load_datasets()

whole_data = np.concatenate((train, test), axis=0)

train1, test1 = train_test_split(whole_data, train_size=0.75, random_state=4)#split dataset into train and test
test2, test_fin = train_test_split(test1, test_size=2500, random_state=4)#split into validation and final test

train = pd.DataFrame(train1, columns = ['DATA_COLUMN','SENTIMENT_COLUMN','LABEL_COLUMN'])
test = pd.DataFrame(test2, columns = ['DATA_COLUMN','SENTIMENT_COLUMN','LABEL_COLUMN'])

DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_SEQ_LENGTH = 128

#Next three functions are used for tokenization and feature generation
def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'DATA_COLUMN', 
                                                                           'LABEL_COLUMN')
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=MAX_SEQ_LENGTH ):
    features = [] 

    for e in examples:
        
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, 
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, 
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)
validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

#creating model
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

start = timer()
history = model.fit(train_data, epochs=2, validation_data=validation_data)
end = timer()
print("Training took: ",timedelta(seconds=end-start))

#plotting training history
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

pred_sentences = test_fin[:, 0]
pred_sentences = pred_sentences.tolist()
test_fin_label = test_fin[:, 2]

tf_batch = tokenizer(pred_sentences, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['Negative', 'Positive']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()


#Confusion matrix and classification report
y_test = np.array(test_fin_label, dtype=np.int64)
y_pred = np.array(label, dtype=np.int64)

#getting predictions label
rows_count = tf_predictions.shape[0]
for i in range (rows_count):
    
    if(tf_predictions[i][0]>tf_predictions[i][1]):
        y_pred[i] = 0
        
    else:
        y_pred[i] = 1
        
print("Confusion matrix: \n\n", confusion_matrix(y_test, y_pred))
print()
print("Classification report: \n", classification_report(y_test, y_pred))


#finding overall polarity for binary classification
def overal_polarity(predictions):
    rows_count = predictions.shape[0]

    neg_count = 0 #count for negative polarity
    pos_count = 0 #count for positive polarity
    for i in range (rows_count):
        if predictions[i] == 0:
            neg_count = neg_count +1

        else:
            pos_count = pos_count + 1

    print("Negative polarity has: ", neg_count)
    print("Positive polarity has: ", pos_count)

    total_sents_count = neg_count + pos_count
    
    if(neg_count)>(1.5*pos_count):
        polarity = "Negative"

    elif(pos_count)>(1.5*neg_count):
        polarity = "Positive"

    else:
        polarity = "Neutral"

    print("Overall polarity is ", polarity)

overal_polarity(label)

#This part of code is the extension of the output of binary classification to 3 class classification

'''
#Using the the ouput from binary claasification to generate 3-class output
def three_classes(pred_weights, label):
    
    #0 maps to Negative polarity
    #1 maps to Neutral
    #2 maps to Positive Polarity
    #pred_weights = tf_predictions
    multi_class_label = label
    rows_count = pred_weights.shape[0]    

    delta = 0.05 #threshold for difference btween negative and positive weights to determine polarity

    for i in range(rows_count):
        #chnging label 1 to 2 for positive polarity
        if multi_class_label[i] == 1:
            multi_class_label[i] = 2

        diff = pred_weights[i][0]-pred_weights[i][1]

        #checking difference   
        if abs(diff)<=delta:
            multi_class_label[i] = 1   
            
    return multi_class_label
            
thr_clas_label = three_classes( tf_predictions, label)


#finding overall polarity for 3-class classification
def thr_clas_overal_polarity(predictions):
    rows_count = predictions.shape[0]

    neg_count = 0 #count for negative polarity
    neut_count = 0 #count for neutral polarity
    pos_count = 0 #count for positive polarity
    
    for i in range (rows_count):
        if predictions[i] == 0:
            neg_count = neg_count +1

        elif predictions[i] == 1:
            neut_count = pos_count + 1
        
        else:
            pos_count =  pos_count + 1

    print("Negative polarity has: ", neg_count)
    print("Neutral polarity has: ", neut_count)
    print("Positive polarity has: ", pos_count)

    total_sents_count = neg_count + pos_count
    
    if (neut_count/total_sents_count)*100 > 0.85:
        polarity = "Neutral"
        
    else:
    
        if(neg_count)>(1.5*pos_count):
            polarity = "Positive"

        elif(pos_count)>(1.5*neg_count):
            polarity = "Negative"

        else:
            polarity = "Neutral"

    print("Overall polarity is ", polarity)
'''