
#!/usr/bin/env python
# coding: utf-8

# This section scrapes the keel website for the relevant text for each of the modules
# Some of the modules do not have valid web pages associated with them so these are handled and exported to a csv file for info.
# 

# In[1]:


import pandas as pd
from tkinter import filedialog
from bs4 import BeautifulSoup
import requests

import_file_path = filedialog.askopenfilename()
df = pd.read_excel (import_file_path)
df = df.iloc[0: , 3:] #exctract module names
x = df.loc[1:1, :].values.flatten().tolist() # Flatten to list
courses = [string[:9].lower() for string in x] #substring and lower case


ModuleDetails = pd.DataFrame( columns=['Module','ModuleTitle','Details'] )
ExcludedModules = pd.DataFrame( columns=['Module','Error'] )

for x in courses:

    URL = "https://www.keele.ac.uk/catalogue/2021-22/"+x+".htm"
    Modulepage = requests.get(URL)
    
    ModuleSoup = BeautifulSoup(Modulepage.content, "html.parser")
    #print(ModuleSoup)
    ModuleName = x
    ModuleAims = ModuleSoup.find_all("div", class_= "col-sm-12")
    ModuleTitle = ModuleSoup.find("div", class_= "panel-heading")
    
    ModuleText = ""
    check = 0
    #print(ModuleTitle)
    for element in ModuleAims:
        if check == 1:
            ModuleText = ModuleText + " " + element.text.strip()
            check = 0    

        if element.text.strip() == "Description for 2021/22":
            check = 1

        if element.text.strip() == "Aims":
            check = 1

        if element.text.strip() == "Intended Learning Outcomes":
            check = 1

    if len(ModuleText)> 0:
        ModuleTitle = ModuleTitle.text.strip()
        new_row = {'Module':ModuleName, 'ModuleTitle':ModuleTitle,'Details':ModuleText}
        ModuleDetails = ModuleDetails.append( new_row, ignore_index=True )
    else:
        new_row = {'Module':ModuleName, 'Error':'Unable to find module details'}
        ExcludedModules = ExcludedModules.append( new_row, ignore_index=True )
        


# In[ ]:


ModuleDetails.to_csv('ModuleDetails.csv')
ExcludedModules.to_csv('Error Report.csv')


# The following section uses the Natural Language Tool Kit to apply Latent Dirichlet Allocation to ascertain the cohesion between the free text in the modules

# In[ ]:


import nltk as nltk

nltk.download('punkt')


# In[ ]:


from tqdm import tqdm_notebook
tqdm_notebook().pandas()


# In[ ]:


ModuleDetails['sentences'] = ModuleDetails.Details.progress_map(nltk.sent_tokenize)


# In[ ]:


ModuleDetails['tokens_sentences'] = ModuleDetails['sentences'].progress_map(lambda sentences: [nltk.word_tokenize(sentence) for sentence in sentences])


# In[ ]:


nltk.download('averaged_perceptron_tagger')


# In[ ]:


ModuleDetails['POS_tokens'] = ModuleDetails['tokens_sentences'].progress_map(lambda tokens_sentences: [nltk.pos_tag(tokens) for tokens in tokens_sentences])


# In[ ]:


from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[ ]:


nltk.download('wordnet')


# In[ ]:


import nltk
nltk.download('omw-1.4')

ModuleDetails['tokens_sentences_lemmatized'] = ModuleDetails['POS_tokens'].progress_map(
    lambda list_tokens_POS: [
        [
            lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
            if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
        ] 
        for tokens_POS in list_tokens_POS
    ]
)


# In[ ]:


nltk.download('stopwords')


# In[ ]:


from nltk.corpus import stopwords

stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
stopwords_other = ['one', 'mr', 'keele', 'student', 'module', 'course', 'graduate', 'caption', 'also', 'foundation', 'something']
my_stopwords = stopwords.words('english') + stopwords_verbs + stopwords_other


# In[ ]:


from itertools import chain # to flatten list of sentences of tokens into list of tokens


# In[ ]:


ModuleDetails['tokens'] = ModuleDetails['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
ModuleDetails['tokens'] = ModuleDetails['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                    and token.lower() not in my_stopwords and len(token)>1])


# In[ ]:


from gensim.models import Phrases


# In[ ]:


tokens = ModuleDetails['tokens'].tolist()
bigram_model = Phrases(tokens)
trigram_model = Phrases(bigram_model[tokens], min_count=1)
tokens = list(trigram_model[bigram_model[tokens]])


# In[ ]:


from gensim import corpora


# In[ ]:


dictionary_LDA = corpora.Dictionary(tokens)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]


# In[ ]:


from gensim import models
import numpy as np


# In[ ]:


np.random.seed(123456)
num_topics = 10
get_ipython().run_line_magic('time', 'lda_model = models.LdaModel(corpus, num_topics=num_topics,                                   id2word=dictionary_LDA,                                   passes=4, alpha=[0.01]*num_topics,                                   eta=[0.01]*len(dictionary_LDA.keys()))')


# In[ ]:


x=lda_model.show_topics(num_topics=num_topics, num_words=20,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

topic_words = []
    
for topic,words in topics_words:
    topic_words.append([topic, str(words)])

Topic_WordsDf = pd.DataFrame(topic_words)
Topic_WordsDf = Topic_WordsDf.rename(columns={0: 'Topic', 1: 'Words'})


# In[ ]:


lda_model[corpus[0]]


# In[ ]:


topics = [lda_model[corpus[i]] for i in range(len(ModuleDetails))]


# In[ ]:


def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res


# In[ ]:


document_topic = pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics])   .reset_index(drop=True).fillna(0)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set(rc={'figure.figsize':(10,20)})
sns.heatmap(document_topic.loc[document_topic.idxmax(axis=1).sort_values().index])


# In[ ]:


ModuleDetails['ModuleID'] = ModuleDetails.index
ModuleDetails.drop(ModuleDetails.iloc[:, 2:7], inplace = True, axis = 1)


# In[ ]:


Coherance = pd.DataFrame(columns=['LDA','ModuleID','Topic','Module','ModuleTitle','tokens','Words'])

for TopicID in range(0, num_topics):
    Topic0 = document_topic[document_topic[TopicID] >=0.5]
    TopicDF = Topic0[TopicID].copy()
    TopicDF = TopicDF.to_frame()
    TopicDF['ModuleID'] = TopicDF.index
    TopicDF['Topic'] = TopicID
    TopicDF = TopicDF.rename(columns={TopicDF.columns[0]: 'LDA'})

    Combined = TopicDF.merge(ModuleDetails)

    Combined = Combined.merge(Topic_WordsDf)
    Coherance = Coherance.append(Combined)
    TopicID += 1


# In[ ]:


Coherance.to_csv('CoherantModules.csv')


# In[ ]:




