#!/usr/bin/env python
# coding: utf-8

# In[7]:


import newspaper
import nltk
#nltk.download('punkt') 
#-- only 1st time
from newspaper import fulltext
import requests
import csv


# In[8]:


cnn_paper = newspaper.build('https://whdh.com/')


# In[9]:


new_dict = {}
for article in cnn_paper.articles:
    article.download()
    article.html
    article.parse()
    title = article.title
    #print(title) 
    #print(article.url) 
    text = article.text
    new_dict[title] = text    
    with open('dict.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in new_dict.items():
            writer.writerow([key, value])
    


    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




