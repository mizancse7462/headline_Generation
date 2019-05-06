#!/usr/bin/env python
# coding: utf-8

# In[65]:


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


# In[66]:
    

#reading csv
train = pd.read_csv('articles_small.csv', encoding='ISO-8859-1',low_memory=False)
#train


# In[67]:


train = train[train.notnull()]
#train


# In[68]:


train = train.dropna(how='any') 
#train


# In[69]:


heads = train['title']
#heads


# In[70]:


descs = train['content']
#descs


# In[71]:


title_list = []
for i in heads:
    title = ftfy.fix_text(i)
    title_list.append(title)  


# In[72]:


title_list


# In[73]:


content_list = []
for i in descs:
    descs = ftfy.fix_text(i)
    content_list.append(descs)


# In[74]:


content_list[0]


# In[75]:



title_list = [''.join(c for c in s if c not in string.punctuation) for s in title_list]


# In[76]:


content_list = [''.join(c for c in s if c not in string.punctuation) for s in content_list]


# In[77]:


tokenized_title = [word_tokenize(i) for i in title_list]


# In[78]:


tokenized_title


# In[79]:


tokenized_content = [word_tokenize(i) for i in content_list]


# In[80]:


tokenized_content[0]


# In[81]:


stop = stopwords.words('english')


# In[82]:


filtered_title = [word for word in tokenized_title if word not in stop]


# In[83]:


filtered_title


# In[84]:


filtered_content = [word for word in tokenized_content if word not in stop]


# In[85]:


filtered_content


# In[86]:


title_new = [' '.join(c for c in s if c not in string.punctuation) for s in filtered_title]


# In[87]:


content_new = [' '.join(c for c in s if c not in string.punctuation) for s in filtered_content]


# In[88]:


title_new


# In[89]:


content_new


# In[90]:


final_list = pd.DataFrame(
    {'heads': title_new,
     'descs': content_new,
    })


# In[91]:


final_list.to_pickle('tokenized_data.pickle')


# In[92]:


df2 = pd.read_pickle('tokenized_data.pickle')
train_data = df2.iloc[:10000]
train_data.to_pickle('train_data.pkl')
validation_data = df2.iloc[10001:13000]
validation_data.to_pickle('validation_data.pkl')
test_data = df2.iloc[13001:14256]
test_data.to_pickle('test_data.pkl')


# In[ ]:





# In[ ]:





# In[ ]:




