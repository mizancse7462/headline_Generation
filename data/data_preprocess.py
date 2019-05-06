#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[16]:


#reading csv
train = pd.read_csv('articles_small.csv', encoding='ISO-8859-1',low_memory=False)
#train


# In[17]:


train = train[train.notnull()]
#train


# In[18]:


train = train.dropna(how='any') 
train


# In[19]:


heads = train['title']
heads


# In[20]:


descs = train['content']
descs


# In[21]:


heads = heads[:50]
descs = descs[:50]


# In[22]:


title_list = []
for i in heads:
    title = ftfy.fix_text(i)
    title_list.append(title)
    #print(title)
    #print('---------------')    


# In[23]:


title_list


# In[24]:


content_list = []
for i in descs:
    descs = ftfy.fix_text(i)
    content_list.append(descs)
    #print(descs)
    #print('---------------')    


# In[25]:


content_list


# In[26]:


data = pd.read_pickle('full_data.pickle')
data


# In[27]:


title = data['heads']
content = data['descs']


# In[ ]:


content


# In[ ]:


title


# In[ ]:



title_list = [''.join(c for c in s if c not in string.punctuation) for s in title]


# In[ ]:


title_list


# In[ ]:


content_list = [''.join(c for c in s if c not in string.punctuation) for s in content]


# In[ ]:


content_list


# In[ ]:


tokenized_title = [word_tokenize(i) for i in title_list]


# In[ ]:


tokenized_title


# In[ ]:


tokenized_content = [word_tokenize(i) for i in content_list]


# In[ ]:


tokenized_content


# In[ ]:


stop = stopwords.words('english')


# In[ ]:


filtered_title = [word for word in tokenized_title if word not in stop]


# In[ ]:


filtered_title


# In[ ]:


filtered_content = [word for word in tokenized_content if word not in stop]


# In[ ]:


filtered_content


# In[ ]:


title_new = [' '.join(c for c in s if c not in string.punctuation) for s in filtered_title]


# In[ ]:


content_new = [' '.join(c for c in s if c not in string.punctuation) for s in filtered_content]


# In[ ]:


title_new


# In[ ]:


content_new


# In[ ]:


final_list = pd.DataFrame(
    {'heads': title_new,
     'descs': content_new,
    })


# In[ ]:


final_list.to_pickle('tokenized_data.pickle')


# In[ ]:


df2 = pd.read_pickle('tokenized_data.pickle')
df2


# In[ ]:


import pickle
favorite_color = pickle.load( open( "tokenized_data.pickle", "rb" ) )


# In[ ]:


favorite_color


# In[ ]:





# In[ ]:





# In[ ]:




