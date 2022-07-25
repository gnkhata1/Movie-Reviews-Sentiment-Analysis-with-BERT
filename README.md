# Movie-Reviews-Sentiment-Analysis-with-BERT
This is the implementation of the Movie Reviews Sentiment Analysis journal version paper. There are four versions of the code but all are typically doing the same thing the difference is just on classification scales and computation of overall polarity from  output vector depending on the number of classes. 

The link to IMDb dataset is: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. Go through Utilities.py to download and use other datasets.

Need to install keras, tensorflow, transformers, pandas, pytorch and pretrained bert. 

Training and testing are done in same file depending on the classification task. We dont include all datasets in the the code, so to use other datasets see Utilities.py code. Run a single file at a time by supplying the follwing command:

_python filename_. e.g:

_python bert-sa.py_

