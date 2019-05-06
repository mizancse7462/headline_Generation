#!/usr/bin/env python
# coding: utf-8


import csv
import os
from collections import defaultdict
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob, Word
#from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.stem import PorterStemmer
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors # load the Stanford GloVe model
import ftfy
import string
from nltk.tokenize import word_tokenize
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')    

#reading csv
train = pd.read_csv('articles_small.csv', encoding='ISO-8859-1',low_memory=False)
#train


train = train[train.notnull()]
#train



train = train.dropna(how='any') 
#train


heads = train['title']
#heads


descs = train['content']
#descs


title_list = []
for i in heads:
    title = ftfy.fix_text(i)
    title_list.append(title)  


title_list




content_list = []
for i in descs:
    descs = ftfy.fix_text(i)
    content_list.append(descs)


content_list[0]


title_list = [''.join(c for c in s if c not in string.punctuation) for s in title_list]


content_list = [''.join(c for c in s if c not in string.punctuation) for s in content_list]


tokenized_title = [word_tokenize(i) for i in title_list]


tokenized_title

tokenized_content = [word_tokenize(i) for i in content_list]

tokenized_content[0]


stop = stopwords.words('english')


filtered_title = [word for word in tokenized_title if word not in stop]


filtered_title


filtered_content = [word for word in tokenized_content if word not in stop]


filtered_content


title_new = [' '.join(c for c in s if c not in string.punctuation) for s in filtered_title]


content_new = [' '.join(c for c in s if c not in string.punctuation) for s in filtered_content]

title_new


content_new


final_list = pd.DataFrame(
    {'heads': title_new,
     'descs': content_new,
    })


final_list.to_pickle('tokenized_data.pickle')


df2 = pd.read_pickle('tokenized_data.pickle')
train_data = df2.iloc[:10000]
train_data.to_pickle('train_data.pkl')
validation_data = df2.iloc[10001:13000]
validation_data.to_pickle('validation_data.pkl')
test_data = df2.iloc[13001:14256]
test_data.to_pickle('test_data.pkl')






