#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[3]:


# Load our data into two Python lists
with open("clickbait.txt") as f:
    lines = f.read().strip().split("\n")
    lines = [line.split("\t") for line in lines]
headlines, labels = zip(*lines)


# In[4]:


headlines[:5]


# In[6]:


# How big is our dataset?
len(headlines)


# In[7]:


# Break dataset into test and train sets
train_headlines = headlines[:8000]
test_headlines = headlines[8000:]

train_labels = labels[:8000]
test_labels = labels[8000:]


# In[8]:


# Create a vectorizer and classifier
vectorizer = TfidfVectorizer()
svm = LinearSVC()


# In[9]:


# Transform our text data into numerical vectors
train_vectors = vectorizer.fit_transform(train_headlines)
test_vectors = vectorizer.transform(test_headlines)


# In[10]:


# Train the classifier and predict on test set
svm.fit(train_vectors, train_labels)


# In[11]:


predictions = svm.predict(test_vectors)
test_headlines[0:5]


# In[12]:


predictions[:5]


# In[13]:


test_labels[:5]


# In[14]:


accuracy_score(test_labels, predictions)


# In[15]:


new_headlines = ["10 Cities That Every Hipster Will Be Moving To Soon", 'Vice President Mike Pence Leaves NFL Game Saying Players Showed "Disrespect" Of Anthem, Flag']
new_vectors = vectorizer.transform(new_headlines)
new_predictions = svm.predict(new_vectors)


# In[16]:


new_predictions


# In[ ]:




