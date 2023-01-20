#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[1]:


## Web-scraping binnenlandsbestuur.nl
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates
from datetime import datetime, timedelta
import locale
import spacy
import nltk
#from nltk.corpus import alpino # Om nederlandse NLTK the gebruiken
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import re
#nltk.download('averaged_perceptron_tagger')
locale.setlocale(locale.LC_ALL, "nl_NL")
#import nltk
nltk.download('alpino')
#from nltk.tag.perceptron import PerceptronTagger


# In[3]:


# Binnenlandsbestuur.nl

paginanummer = 0
aantal_dagen = 7
huidige_dag = datetime.today().date()
max_toegestane_afwijking_huidige_dag = timedelta(days=aantal_dagen) # Even wat korter zodat code sneller runt 
url = "https://www.binnenlandsbestuur.nl/digitaal/nieuws?page=0"  # basis URL
verzameling_urls = [] # Om URLs uit loop in op te slaan
datums_df = [] # Om datums van publicatie artikelen in op te slaan
klaar = False
# Haal met een loop automatisch alle relevante artikelen op
while True: # run loop terwijl IF statement TRUE is
    page = requests.get(url) # benader URL
    soup = BeautifulSoup(page.content, 'html.parser') # download HTML-code
    artikelen_op_pagina = soup.find_all("a", {"class": "o-block-link"}) # Zoek HTML blokken met informatie over artikelen
    datums_op_pagina = soup.find_all('div', {"class": "c-meta__item c-meta__item--publicationDate"}) # Zoek datums van artikelen op
    datums = []
    for datum in datums_op_pagina:
        datums.append(datetime.strptime(datum.find_all("span", {"class": "c-meta__text"})[0].text, '%d %B %Y' ).date()) # Datums die horen bij artikelen op pagina
    for ind in range(0,len(datums)):
        if ind == len(datums)-1:
            paginanummer +=1 # We kunnen door naar de volgende pagina met nieuwe artikelen
            url = "https://www.binnenlandsbestuur.nl/digitaal/nieuws?page="+str(paginanummer)
        if huidige_dag - datums[ind] < max_toegestane_afwijking_huidige_dag:
            verzameling_urls.append("https://www.binnenlandsbestuur.nl"+artikelen_op_pagina[ind].get('href')) # URLs van de verschillende artikelen op pagina  
            datums_df.append(datums[ind]) # datum van het artikel
        else:
            klaar = True
            break
    if klaar == True:
        break

# Nu gaan we alle verzamelde artikelen los doornemen        
titels = []
tekst = []        

for url in verzameling_urls:    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser') # download HTML-code
    # Pak eerst de header. Deze bevat de titel en een kort stukje tekst
    header_blok = soup.find_all("div", {"class": "c-article-header"})
    header_titel = header_blok[0].find_all("h1", {"class": "c-article-header__heading"})[0].text
    header_tekst = header_blok[0].find_all("p", {"class": "c-article-header__lead"})[0].text
    # Haal nu de rest van de tekst op
    hoofd_artikel = soup.find("main", {"class": "b-article__main"})
    try:
        hoofd_artikel_tekst = hoofd_artikel.find("div", {"class": "c-article-content s-article"}).text
    except: 
        hoofd_artikel_tekst = hoofd_artikel.find("div", {"class": "c-article-content has--blockquote s-article"}).text
    titels.append(header_titel)
    tekst.append(header_tekst+str("\n")+hoofd_artikel_tekst)
    
    
# Verzamel titel, tekst en datum netjes in dataframe
df = pd.DataFrame({"titel": titels, "tekst": tekst, "datum": datums_df, "url": verzameling_urls})

# Extra stap: Beschouw alleen de artikelen onder de subkop 'Digitaal'. Dit is makkelijker hier te doen dan in webscraper
df_binnenlands = df[df["url"].str.contains('digitaal')].reset_index()


# In[4]:


# vng.nl
#aantal_dagen = 183
paginanummer = 0
huidige_dag = datetime.today().date()
max_toegestane_afwijking_huidige_dag = timedelta(days=aantal_dagen) # komt uit cell hierboven!
url = "https://vng.nl/nieuws?sort_bef_combine=created_DESC&sort_by=created&sort_order=DESC&page=0"  # basis URL
verzameling_urls = [] # Om URLs uit loop in op te slaan
datums_df = [] # Om datums van publicatie artikelen in op te slaan
klaar = False
# Haal met een loop automatisch alle relevante artikelen op
while True: # run loop terwijl IF statement TRUE is
    page = requests.get(url) # benader URL
    soup = BeautifulSoup(page.content, 'html.parser') # download HTML-code
    artikelen_op_pagina = soup.find_all("div", {"class": "layout__region layout__region--second node-list__item-column--content"}) # Zoek HTML blokken met informatie over artikelen
    datums_op_pagina = soup.find_all('div', {"class": "field field--label-hidden field--node-post-date"}) # Zoek datums van artikelen op
    datums = []
    for datum in range(0,len(datums_op_pagina)):
        datums.append(datetime.strptime(re.sub('\s+',' ',datums_op_pagina[datum].text).strip(), '%d %B %Y').date()) # Datums die horen bij artikelen op pagina
    for ind in range(0,len(datums)):
        if ind == len(datums)-1:
            paginanummer +=1 # We kunnen door naar de volgende pagina met nieuwe artikelen
            url = "https://vng.nl/nieuws?sort_bef_combine=created_DESC&sort_by=created&sort_order=DESC&page="+str(paginanummer)
        if huidige_dag - datums[ind] < max_toegestane_afwijking_huidige_dag:
            verzameling_urls.append("https://www.vng.nl"+artikelen_op_pagina[ind].find('a')['href']) # URLs van de verschillende artikelen op pagina  
            datums_df.append(datums[ind]) # datum van het artikel
        else:
            klaar = True
            break
    if klaar == True:
        break   

# Nu gaan we alle verzamelde artikelen los doornemen        
titels = []
tekst = []            
    
for url in verzameling_urls:    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser') # download HTML-code
    # Pak eerst de header. Deze bevat de titel en een kort stukje tekst
    #header_blok = soup.find_all("div", {"class": "c-article-header"})
    header_titel = re.sub('\s+',' ',soup.find("h1", {"class": "page-title"}).text).strip()
    #header_tekst = header_blok[0].find_all("p", {"class": "c-article-header__lead"})[0].text
    # Haal nu de rest van de tekst op
    hoofd_artikel_tekst = re.sub('\s+', ' ',soup.find("div", {"class": "l-main-content"}).text)
    titels.append(header_titel)
    tekst.append(hoofd_artikel_tekst)
    
# Verzamel titel, tekst en datum netjes in dataframe
df_vng = pd.DataFrame({"titel": titels, "tekst": tekst, "datum": datums_df, "url": verzameling_urls})

# Extra stap: Beschouw alleen de artikelen onder de subkop 'Digitaal'. Dit is makkelijker hier te doen dan in webscraper
#df = df[df["url"].str.contains('digitaal')].reset_index()  


# In[5]:


# digitaleoverheid.nl
#aantal_dagen = 183
paginanummer = 1
huidige_dag = datetime.today().date()
max_toegestane_afwijking_huidige_dag = timedelta(days=aantal_dagen) # komt uit cell hierboven!
url = "https://www.digitaleoverheid.nl/actueel/nieuws/"  # basis URL
verzameling_urls = [] # Om URLs uit loop in op te slaan
datums_df = [] # Om datums van publicatie artikelen in op te slaan
klaar = False
# Haal met een loop automatisch alle relevante artikelen op
while True: # run loop terwijl IF statement TRUE is
    page = requests.get(url) # benader URL
    soup = BeautifulSoup(page.content, 'html.parser') # download HTML-code
    artikelen_op_pagina = soup.find_all("div", {"class": "txtcontainer"}) # Zoek HTML blokken met informatie over artikelen
    datums_op_pagina = soup.find_all('div', {"class": "field field--label-hidden field--node-post-date"}) # Zoek datums van artikelen op
    datums = []
    for datum in range(0,len(artikelen_op_pagina)):
        datums.append(datetime.strptime(artikelen_op_pagina[datum].find('p').text[16:], '%d %B %Y').date()) # Datums die horen bij artikelen op pagina
    for ind in range(0,len(datums)):
        if ind == len(datums)-1:
            paginanummer +=1 # We kunnen door naar de volgende pagina met nieuwe artikelen
            url = "https://www.digitaleoverheid.nl/actueel/nieuws/page/"+str(paginanummer)+"/"
        if huidige_dag - datums[ind] < max_toegestane_afwijking_huidige_dag:
            verzameling_urls.append(artikelen_op_pagina[ind].find('a')['href']) # URLs van de verschillende artikelen op pagina  
            datums_df.append(datums[ind]) # datum van het artikel
        else:
            klaar = True
            break
    if klaar == True:
        break   

# Nu gaan we alle verzamelde artikelen los doornemen        
titels = []
tekst = []            
    
for url in verzameling_urls:    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser') # download HTML-code
    # Pak eerst de header. Deze bevat de titel en een kort stukje tekst
    #header_blok = soup.find_all("div", {"class": "c-article-header"})
    header_titel = re.sub('\s+', ' ',soup.find("h1").text).strip()
    #header_tekst = header_blok[0].find_all("p", {"class": "c-article-header__lead"})[0].text
    # Haal nu de rest van de tekst op
    hoofd_artikel_tekst = re.sub('\s+', ' ',soup.find("header", {"class": "entry-header"}).text)
    titels.append(header_titel)
    tekst.append(hoofd_artikel_tekst)
    
# Verzamel titel, tekst en datum netjes in dataframe
df_digitale_overheid = pd.DataFrame({"titel": titels, "tekst": tekst, "datum": datums_df, "url": verzameling_urls})

# Extra stap: Beschouw alleen de artikelen onder de subkop 'Digitaal'. Dit is makkelijker hier te doen dan in webscraper
#df = df[df["url"].str.contains('digitaal')].reset_index()


# In[6]:


#samenvoegen dataframes
df_binnenlands["bron"] = "binnenlandsbestuur"
df_vng["bron"] = "vng"
df_digitale_overheid["bron"] = "digitale overheid"

df = pd.concat([df_binnenlands,df_vng,df_digitale_overheid], ignore_index=True, axis=0)
df.drop('index',inplace = True, axis = 1)



# In[7]:


# Leestekens verwijderen, hoofdletters vervangen door kleine letters en titel en tekst samenvoegen
df['schone_titel'] = df['titel'].str.replace('[^A-Za-z0-9 ]+', ' ').str.lower()
df['schone_tekst'] = df['tekst'].astype(str).str.replace('[^A-Za-z0-9 ]+', ' ').str.lower()
df['titel_plus_tekst'] = df['schone_titel'] + df['schone_tekst']

# Inladen van dataset met positieve en negatieve woorden
# (Chen, Y., & Skiena, S., 2014. Building Sentiment Lexicons for All Major Languages. In ACL (2) (pp. 383-389))
negatief = pd.read_csv("C:/Users/WesselJansenvanDoorn/OneDrive - InIn Advies B.V/Topic Analysis/negative_words.txt", header = None).rename(columns={0: 'woord'})
positief = pd.read_csv("C:/Users/WesselJansenvanDoorn/OneDrive - InIn Advies B.V/Topic Analysis/positive_words.txt", header = None).rename(columns={0: 'woord'})
negatief["negatief"] = 1
positief["positief"] = 1
woorden = positief.append(negatief).fillna(0) #1 Grote dataframe met woorden en bijbehorend sentiment

## We willen gaan kijken hoeveel teksten overwegend positief en overwegend negatief zijn

# Stap 1: Tekst opsplitsen in woorden
df_sentiment = df[['titel_plus_tekst']]
df_sent = df_sentiment['titel_plus_tekst'].str.split(' ', expand=True).stack().reset_index(level=0)
df_sent = df_sent.rename(columns={0: 'woord'})

# Stap 2: Merge de dataframe met losse woorden (df_sent) met onze dataframe met woorden
join = df_sent.merge(woorden, how='inner', on='woord')
join_sum = join.groupby('level_0')[["positief", "negatief"]].sum() # breng weer terug naar artikelniveau: hoeveel positieve en negatieve woorden staan er in artikel?

# Stap 3: Voeg alles weer samen in een grote dataframe
data = df.merge(join_sum, left_index=True, right_index=True, how='left')

# Stap 4: Bereken het verschil positief en negatief; is een tekst overwegend positief of negatief?
#df['verschil'] = data['positief'] - data['negatief']

# Stap 5: Splits de artikelen op in artikelen die gaan over AI en de rest
#df['split'] = 0
#df.loc[df['titel_plus_tekst'].str.contains("ai"), 'split'] = 1                      #\ kan bijvoorbeeld ook alleen te titel bekijken
#df.loc[df['titel_plus_tekst'].str.contains("artificial intelligence"), 'split'] = 1 #/

# Plot het verschil tussen positief en negatief
#plt.scatter(df['datum'].astype('str'), df['verschil'], c = df['split'], cmap='viridis', alpha=0.5)
#plt.scatter(dates.date2num(df['datum'].tolist()), df['verschil'], c = df['split'], cmap='viridis', alpha=0.5)
#plt.axhline(y=0, linewidth=2, color='black')
#plt.colorbar(drawedges = True, label = 'split')
#plt.xlabel('datum')
#plt.ylabel('sentimentscore')
#plt.xticks(dates.date2num(df['datum'].tolist()),df['datum'].tolist(), rotation = 45)
#plt.locator_params(nbins=4)
#plt.title('sentiment tegenover Artificial Intelligence in verschillende artikelen')

# Vgm gaan datums toch nog niet helemaal goed?


# Om bovenstaande sentiment analyse te kunnen gebruiken hebben we duidelijke topics nodig.

# ## Eerste visualisatie topics: WordCloud

# In[8]:


#from wordcloud import WordCloud
# Maak kolom voor titels zonder stopwoorden
dutch_stopwords = pd.read_csv("C:/Users/WesselJansenvanDoorn/OneDrive - InIn Advies B.V/Topic Analysis/stopwoorden.txt", header = None).rename(columns={0: 'woord'})
dutch_stopwords = dutch_stopwords["woord"].tolist()
df["schone_titel_zonder_stopwords"] = df["schone_titel"].apply(lambda x: ' '.join([word for word in x.split() if word not in (dutch_stopwords)]))
df["schone_titel_plus_tekst_zonder_stopwords"] = df["titel_plus_tekst"].apply(lambda x: ' '.join([word for word in x.split() if word not in (dutch_stopwords)]))

# Join de verschillende titels
#long_string = ",".join(list(df["schone_titel_zonder_stopwords"].values)) 
#long_string2 = ",".join(list(df["schone_titel_plus_tekst_zonder_stopwords"].values)) 

# Maak een WordCloud object
#wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
#wordcloud2 = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Genereer een word cloud
#wordcloud.generate(long_string)
#wordcloud2.generate(long_string2)


# In[9]:


# Visualiseer wordcloud voor alleen de titels
#wordcloud.to_image()


# In[10]:


# Visualiseer wordcloud voor de gehele tekst
#wordcloud2.to_image()


# ## Topic Analysis: LDA - preprocessing.
# Als hoofvorm voor LDA gebruiken we een aanpak met gensim. Dit omdat we hiermee meer vrijheid hebben binnen het gebruik van bestaande implementaties. Met een package als sk-learn kunnen we bijvoorbeeld moeilijker pre-processen en optimizen.
# 
# Onderstaande code maakt de tekst die we hebben klaar voor LDA. Hierbij focussen we op lemmatization, bi- en trigrams en het behouden van (zo veel mogelijk) alleen zelfstandige naamwoorden (omdat die iets zeggen over topics).

# In[9]:


# Lemmatization

# Initialize spacy 'nl_core_news_sm' model, keeping only tagger component needed for lemmatization
nlp = spacy.load("nl_core_news_sm", disable=['parser', 'ner'])

# Voer Lemmatization uit op onze dataframe
df["lematized_tekst_en_titel_zonder_stopwords"] = df["schone_titel_plus_tekst_zonder_stopwords"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))


# In[10]:


# Bi-grams en Tri-grams
# Tekst als 'artificial intelligence' zou eigenlijk gezien moeten worden als één woord

#Bi-gram
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_documents([comment.split() for comment in df["lematized_tekst_en_titel_zonder_stopwords"]])
# Filter only those that occur at least 50 times
finder.apply_freq_filter(3) # Werkt wel om zaken als "artificial intelligence" te vinden. Maar corpus is nu zo klein dat amper voorkomt
bigram_scores = finder.score_ngrams(bigram_measures.pmi)

#Tri-gram
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = nltk.collocations.TrigramCollocationFinder.from_documents([comment.split() for comment in df["lematized_tekst_en_titel_zonder_stopwords"]])
# Filter only those that occur at least 50 times
finder.apply_freq_filter(3)
trigram_scores = finder.score_ngrams(trigram_measures.pmi)


# In[11]:


bigram_pmi = pd.DataFrame(bigram_scores)
bigram_pmi.columns = ['bigram', 'pmi']
bigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

trigram_pmi = pd.DataFrame(trigram_scores)
trigram_pmi.columns = ['trigram', 'pmi']
trigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)


# In[12]:


# Filter for bigrams with only noun-type structures
stop_word_list = dutch_stopwords
def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if bigram[0] in stop_word_list or bigram[1] in stop_word_list:
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    if 'PRON' in bigram:
        return False
    return True

# Filter for trigrams with only noun-type structures
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if trigram[0] in stop_word_list or trigram[-1] in stop_word_list or trigram[1] in stop_word_list:
        return False
    if 'n' in trigram or 't' in trigram:
         return False
    if 'PRON' in trigram:
        return False
    return True 


# In[13]:


# Can set pmi threshold to whatever makes sense - eyeball through and select threshold where n-grams stop making sense
# choose top 500 ngrams in this case ranked by PMI that have noun like structures
filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram:                                              bigram_filter(bigram['bigram'])                                              and bigram.pmi > 5, axis = 1)][:500]

filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram:                                                  trigram_filter(trigram['trigram'])                                                 and trigram.pmi > 5, axis = 1)][:500]


bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]


# In[14]:


# Eigen functie: Vervang 'Artificial intelligence" & "kunstmatige intelligentie" door 'ai'
te_vervangen = ["artificial intelligence", "kunstmatige intelligentie"]

def replace_ai(x):
    for word in te_vervangen:
        x = x.replace(word, "ai")
    return x


# In[15]:


# Concatenate n-grams
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x


# In[16]:


tekst_met_ngrams = df["lematized_tekst_en_titel_zonder_stopwords"].copy()
tekst_met_ngrams.reviewText = tekst_met_ngrams.map(lambda x: replace_ai(x))
tekst_met_ngrams.reviewText = tekst_met_ngrams.reviewText.map(lambda x: replace_ngram(x))


# In[17]:


tekst_met_ngrams = tekst_met_ngrams.reviewText.map(lambda x: [word for word in x.split()                                                 if word not in stop_word_list                                                              and word not in dutch_stopwords                                                              and len(word) > 1])


# In[18]:


# Filter op alleen zelfstandige naamwoorden
# Topics zijn vaak zelfstandige naamwoorden. Door alleen deze te behouden heeft ons topic model het makkelijker
# Filter for only nouns
def noun_only(x):
    #pos_comment = nltk.pos_tag(x, lang = 'nld')
    pos_comment = nltk.pos_tag(x, lang = 'eng') #vervangen door spacy dutch tagger?
    filtered = [word[0] for word in pos_comment if word[1] in ['NN'] or word[1] in ['NP'] or word[1] in ['VERB']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered

uiteindelijke_tekst = tekst_met_ngrams.map(noun_only)


# In[19]:


def noun_only(x):
    doc = nlp(' '.join(str(e) for e in x))
    filtered = []
    for token in doc:
        if token.pos_ == 'NOUN': #bewaar alleen zelfstandige naamwoorden
            filtered.append(token.orth_)
    return filtered

uiteindelijke_tekst = tekst_met_ngrams.map(noun_only)


# In[20]:


uiteindelijke_tekst


# In[21]:


#token,pos = [],[]
#for sent in nlp.pipe(tekst_met_ngrams[0]):
#    if sent.has_annotation('DEP'):
#        #add the tokens present in the sentence to the token list
#        token.append([word.text for word in sent])
#        #add the pos tage for each token to the pos list
#        pos.append([word.pos_ for word in sent])


# In[22]:


#doc = nlp('honden en zwijnen gaan rennen')
#filtered = []
#for token in doc:
#    if token.pos_ == 'NOUN':
#        filtered.append(token.orth_)


# In[81]:





# ## LDA - Analyse

# In[23]:


dictionary = corpora.Dictionary(uiteindelijke_tekst)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in uiteindelijke_tekst]


# In[24]:


n = 50
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=n, id2word = dictionary, passes=40,               iterations=200,  chunksize = 10000, eval_every = None, random_state=0)


# In[25]:


ldamodel.show_topics(n, num_words=10, formatted=False)


# In[99]:


# Visualisatie
topic_data =  pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary, mds = "mmds")#mds = 'pcoa')
pyLDAvis.save_html(topic_data, 'lda.html')
pyLDAvis.display(topic_data)





# In[27]:


# Voeg onderwerpen toe aan dataframe
all_topics = ldamodel.get_document_topics(doc_term_matrix, minimum_probability=0.0)
all_topics_csr = gensim.matutils.corpus2csc(all_topics)
all_topics_numpy = all_topics_csr.T.toarray()
all_topics_df = pd.DataFrame(all_topics_numpy)
df["onderwerp"]= [np.argmax(topic) for topic in all_topics_numpy]


# In[101]:


df


# ## Wegschrijven data

# In[28]:


df_wegschrijven = df[["titel", "datum", "url", "onderwerp"]]
top_words_per_topic = []
for t in range(ldamodel.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in ldamodel.show_topic(t, topn = 10)])

df_topics_wegschrijven = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])

# with pd.ExcelWriter(f'outputLDAlaatste' +str(aantal_dagen) + "dagen" + str(datetime.today().date())+'.csv') as writer:  
#     df_topics_wegschrijven.to_csv(writer, sheet_name='Topics')
#     df_wegschrijven.to_csv(writer, sheet_name='Artikelen en onderwerpen')


# In[29]:


df_topics_wegschrijven


# In[57]:


df_topics_wegschrijven.to_csv("dftopicswegschrijven.csv")
uiteindelijke_tekst.to_csv("uiteindelijke_tekst.csv") #inlezen met test =  pd.read_csv("uiteindelijke_tekst.csv", index_col = 0, squeeze = True)


