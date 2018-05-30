import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim

import os, re, operator, warnings
warnings.filterwarnings('ignore')

#IPython kernel works seamlessly with matplotlib by showing all the plots inline in the Jupyter notebook.
import requests
r = requests.get('https://techcrunch.com/')

from bs4 import BeautifulSoup
soup = BeautifulSoup(r.text, 'html.parser') #read the html and make sense of its structure
                                            #parser included by python library, though other can also be used

    
results1 = soup.find_all('a', attrs= {'class': 'post-block__title__link'})
results2 = soup.find_all('div', attrs= {'class': 'post-block__content'})

records_title = []

for result in results1:
    pandey = result.contents[0]
    updated_pandey = ' '.join(pandey.split())
    records_title.append((updated_pandey))


records_body = []

for result in results2:
    bishal = result.contents[0]
    updated_bishal = ' '.join(bishal.split())
    records_body.append((updated_bishal))

records_body_string = " ".join(str(x) for x in records_body)
records_title_string = " ".join(str(x) for x in records_title)
records_combined = records_body_string + "." +  records_title_string
records_combined

from spacy.lang.en import English
#creating english language pipeline
#pipeline consists of a chain of processing elements (processes, threads, coroutines, functions, etc.), arranged so that the output of each element is the input of the next; the name is by analogy to a physical pipeline.
nlp = spacy.load("en")#this gave me trouble so I used python -m spacy download en
#this is spacy english language model, which consist of built in english stopwords
my_stop_words = ['say', 'Mr', 'be', 'said', 'says', 'saying', 'its', 'it', '’s', 'the', 'there', 't', 'th']

for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword] #applying spacy pipeline to my stopword
    lexeme.is_stop = True #and making sure that its a stopword

doc = nlp(records_combined)

texts, articles = [], []

for w in doc:
    #if it is not a stop word or punctuation mark, add it to our article
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        #we add the lematized version of the word, root of the word
        articles.append(w.lemma_)

#captain_raj = [x.encode('UTF8') for x in articles]#this is gensim corpus
#captain_raj

shyam = []

phrases = gensim.models.Phrases(articles)
bigram = gensim.models.phrases.Phraser(phrases)
captain_rajs = bigram[articles]  
list_de_list = [[captain_raj] for captain_raj in captain_rajs]
print(list_de_list)

dictionary = Dictionary(list_de_list)
corpus = [dictionary.doc2bow(list_de_list) for list_de_list in list_de_list]

ldamodel = LdaModel(corpus = corpus, num_topics = 2, id2word = dictionary)

visualisation = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html') 


