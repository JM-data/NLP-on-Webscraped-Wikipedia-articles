
# coding: utf-8

# ## Text extraction

# Let's webscraped some wikipedia sites having different languages and then cluster them ! We use here the three languages English, French and Spanish. Different stopwords lists are used, same for stemming, since there are 3 languages.

# In[18]:


import wikipedia
import random
from random_words import RandomWords
import pandas as pd

rw = RandomWords()
lang = ['en','fr','es']


# ###### Building a Data base like that

# We first start with the wikipedia page with Phone numbers, since the tasks asks to get phone numbers. Sometimes, the random word chosen isn't on Wikipedia, so we have to raise exceptions.

# In[19]:


Index = ['Telephone number']
Text = []
Language = ['en']

wikipedia.set_lang('en')
word = 'Telephone number'
Sentence = wikipedia.page(word).content.replace('\n', '')
Text.append(Sentence)

s = 0
while s<49:
    while True:
        language = random.choice(lang)
        wikipedia.set_lang(language)
        word = rw.random_word()
        try:
            Sentence = wikipedia.summary(word, sentences=4)
            Index.append(word)
            Text.append(Sentence)
            Language.append(language) 
            s+=1
            break
        except wikipedia.exceptions.DisambiguationError:
            pass
        except wikipedia.exceptions.PageError:
            pass


# In[288]:


dic = {'Text': Text,
      'Language': Language,
      'Word' : Index}

df = pd.DataFrame.from_dict(dic)


# In[289]:


print(df)


# ## Find phone numbers

# In[290]:


import re

telephone_numbers = []


# In[291]:


for i in range(len(df)):
    phone_numbers = re.findall('(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})',
                               df.loc[i,'Text'])
    telephone_numbers.append(phone_numbers)
    if phone_numbers:
        print("The phone numbers found on the wikipedia page '%s' :"%df.loc[i,'Word'])
        for w in phone_numbers:
            print(w)
            
df['Phone Numbers found'] = telephone_numbers


# ## Determining the language

# Using Google's language detection library

# In[292]:


from langdetect import detect
lang_detect = []
Test = []


# In[293]:


for i in range(len(df)):
    detected_language = detect(df.loc[i,'Text'])
    lang_detect.append(detected_language)
    test = detect(df.loc[i,'Text']) == df.loc[i,'Language']
    Test.append(test)
    print('Real Language : %s'%detected_language)
    print('Detected Languages : %s'%df.loc[i,'Language'])
    print('True if both languages are the same : %s' %test)
    print('\n')

df['Language detected'] = lang_detect
df['Same language ?'] = Test


# ## Part of speech tagging

# In[294]:


from nltk.tag import pos_tag

proper_nouns = []


# In[295]:


for i in range(len(df)):
    text = df.loc[i,'Text']
    tagged_sent = pos_tag(text.split(), lang=df.loc[i,'Language'])  
    propernouns = [word.replace("«",'').replace("»",'').replace('==','').replace('=','') for word,pos in tagged_sent if pos == 'NNP']
    print(*propernouns[:5], sep='\n')
    proper_nouns.append(propernouns)
    
df['Proper Nouns'] = [' '.join(liste) for liste in proper_nouns]


# In[296]:


df.tail()


# Careful for the French NNP !

# ## Remove irrelevant words

# In[297]:


from nltk.corpus import stopwords

stopwords_en = stopwords.words('english')
stopwords_fr = stopwords.words('french')
stopwords_es = stopwords.words('spanish')

Filtered = []


# In[298]:


for i in range(len(df)):
    text = df.loc[i,'Text'] 
    filtered = [word.replace("«",'').replace("»",'').replace('==','') for word in text.split() if word.lower() not in globals()['stopwords_'+df.loc[i,'Language']]]
    Filtered.append(filtered)
    
df['Filtered'] = [' '.join(liste) for liste in Filtered]


# In[299]:


print(df)


# ## Shrink the vector space

# In[300]:


from nltk.stem import SnowballStemmer
from nltk.stem.snowball import FrenchStemmer

stemmer_en = SnowballStemmer('english')
stemmer_fr = FrenchStemmer()
stemmer_es = SnowballStemmer('spanish')

Stemmed = []


# In[301]:


for i in range(len(df)):
    text = df.loc[i,'Filtered'] 
    stemmed = [globals()['stemmer_'+df.loc[i,'Language']].stem(word) for word in text.split()]
    Stemmed.append(stemmed)
    print('#'*75)
    print('The word %s is in the %s language' %(df.loc[i,'Word'],df.loc[i,'Language']))
    print(*stemmed[:5], sep = '\n')
    
df['Stemmed'] = [' '.join(liste) for liste in Stemmed]


# ## Cluster documents into logical groups

# In[275]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
 
documents = df['Stemmed']
 

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
 
true_k = 8
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
 
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind])

Cluster = []

for i in range(len(df)):
    Y = vectorizer.transform([df.loc[i,'Stemmed']])
    prediction = model.predict(Y)
    Cluster.append(prediction)
    #print(df.loc[i,'Word'],prediction)

df['Cluster'] = Cluster


# In[276]:


df.head()


# In[283]:


print(df[df['Cluster'] == 5])


# ## Produce a basic analysis of the result

# Interesting : the words might be random, but the KNN discovered a cluster, with different languages on top of it, that has to do with climate, environment. The words we find are skiers, canals and meat.

# Then there's a more physics related cluster. Having words like rains, vapor, canals and wave.

# ## Summary

# Hard to cluster Wikipedia in different articles. Not all the climat things were in the right cluster.
# 
# Since this code will each time get random words, I hope this python jupyter notebook shows the analysis on the set of Data webscrapped in the first place.
