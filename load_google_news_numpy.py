#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import gensim
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True,limit=10 ** 4)


# In[ ]:


print([method for method in dir(model)])


# In[ ]:


print(len(model.vectors), model.vector_size)


# In[ ]:


def make_dataset(model):
    """Make dataset from pre-trained Word2Vec model.
    Paramters
    ---------
    model: gensim.models.word2vec.Word2Vec
        pre-traind Word2Vec model as gensim object.
    Returns
    -------
    numpy.ndarray((vocabrary size, vector size))
        Sikitlearn's X format.
    """
    V = model.index2word
    X = np.zeros((len(V), model.vector_size))

    for index, word in enumerate(V):
        X[index, :] += model[word]
    return X


# In[ ]:


google_list = make_dataset(model)


# In[ ]:


print(google_list)


# In[ ]:


import numpy
numpy.save('google-news-nump.npy',google_list)


# In[ ]:


for i,k in enumerate(model.vocab):
    if(i > 10):
        break
    print(k)


# In[ ]:




