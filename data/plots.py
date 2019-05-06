#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import os
from collections import defaultdict
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob, Word
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors # load the Stanford GloVe model
import ftfy
import string
from nltk.tokenize import word_tokenize
import pickle
import nltk
nltk.download('punkt')


# In[ ]:


df2 = pd.read_pickle('tokenized_data.pickle')


# In[ ]:


heads = df2['heads']
heads


# In[ ]:


descs = df2['descs']
descs


# In[ ]:


i=0
heads[i]


# In[ ]:


descs[i]


# In[ ]:


len(heads),len(set(heads))


# In[ ]:


len(descs),len(set(descs))


# In[ ]:


from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount


# In[ ]:


vocab, vocabcount = get_vocab(heads+descs)


# ### Most popular tokens 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot([vocabcount[w] for w in vocab]);
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log", nonposy='clip')
plt.title('word distribution in headlines and discription')
plt.xlabel('rank')
plt.ylabel('total appearances');


# In[ ]:





# In[ ]:




