# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:43:23 2022

@author: wicky\
"""

#%%
import re,string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
import pickle
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from tqdm import tqdm
from sklearn import utils
tqdm.pandas(desc="progress-bar")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor

from sklearn.manifold import MDS

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

import pandas as pd
import os
import optuna
from functools import partial
import gensim
from gensim.models import Word2Vec,LdaMulticore, TfidfModel
from gensim import corpora
from gensim import models

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import numpy as np

#%%
os.chdir(r"C:\Users\wicky\OneDrive\Documents\Northwestern DSMS\MSDS453\Project")
# read data

df = pd.read_csv("winemag-data_first150k.csv", index_col=(0))

df.isna().sum()

#Cleaning Data

df.dropna(inplace = True, subset=("price", "region_1")) #dropping rows that does not have a price or region
df.drop_duplicates(subset=("description"), inplace= True)
df.reset_index(drop = True, inplace= True)

df = df.loc[df["price"] <= 150] #dropping wine price greater than $150, too expensive for us to afford

col = df.columns
#%%
#EDA
sns.set(rc = {'figure.figsize':(15,8)})
sns.boxplot(x= "price", y="country", data = df, order=(["Canada", "US", "France", "Italy", "Australia",
                                                        "Spain", "Argentina"]))
# Canada makes the most expensive wines by median
# Canada's wine price seems to be the most consistent

sns.boxplot(x= "points", y="country", data = df)
# Canada's wine score rank the highest overall

wine_under_20usd = df.loc[df["price"] <= 20] #honestly $20 is probably the most I would pay for wine (if it's in Costco or a grochery store)

sns.scatterplot(data = wine_under_20usd, x = "price", y="points")# can we find a cheap wine with good rating?

celebration_wine = df.loc[df["points"] >= 99].sort_values("price", ascending= True) #for celebrations! Expensive, but the best ratings

date_wine = wine_under_20usd.loc[wine_under_20usd["points"] >= 94].sort_values("points", ascending= False) #for date nights, little more expensive, but worth the money

daily_wine = wine_under_20usd.loc[wine_under_20usd["points"] >= 92].sort_values("price", ascending= True) #to save money, sort by price but >= 92 points



#%%
#Analyzing the reviews on high_rating_wine, greating than 96 points
def clean_doc(doc): 
    #split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # word stemming    
    #ps=PorterStemmer()
    #tokens=[ps.stem(word) for word in tokens]
    # filter out stop words
    stop_words = stopwords.words('english')
    stop_words.extend(["flavor","flavors", "drink", "fruit", "finish", "tanning"]) #extending stop_words after reviewing WordClouds
    tokens = [w for w in tokens if not w in stop_words]         
    
    return tokens

df["tokens"] = df["description"].apply(lambda x: clean_doc(x)) #create tokens
df["cleaned"] = df["tokens"].apply(lambda x: str(" ".join(x))) #stitching

high_rating_wine = df.loc[df["points"] > 95]
high_rating_wine.reset_index(drop = True, inplace= True)
low_rating_wine = df.loc[df["points"] <= 83]
low_rating_wine.reset_index(drop = True, inplace= True)

#Starting with one review:
text = high_rating_wine["cleaned"][0]
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Joining all high scoring wine reviews to one giant string
text = " ".join(review for review in high_rating_wine["cleaned"])
print ("There are {} words in the combination of all high_rating_wine review.".format(len(text)))
wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.title("High Rating Wine WordCloud (>96 points)")
plt.axis("off")
plt.show()


#Joining all low scoring wine reviews to one giant string
text = " ".join(review for review in low_rating_wine["cleaned"])
print ("There are {} words in the combination of all low_rating_wine review.".format(len(text)))
wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Low Rating Wine WordCloud (<=55 points)")
plt.axis("off")
plt.show()

#%%
#Create df2 for sampling, too many data points
df2 = df.sample(frac=.15, replace=False, random_state = 99)
df2 = df2.reset_index(drop = True)

doc = df2["cleaned"].tolist() #getting ready for TF-IDF

#TF-IDF
Tfidf=TfidfVectorizer(ngram_range=(1,2), max_features=(100))

TFIDF_matrix=Tfidf.fit_transform(doc)

matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names())


weight = TFIDF_matrix.toarray()
X = weight



#%% LDA Topic Modeling

id2word = corpora.Dictionary(df2["tokens"])
dictionary = corpora.Dictionary(df2["tokens"])
corpus = [dictionary.doc2bow(doc) for doc in df2["tokens"]]
lda_model = gensim.models.LdaMulticore(corpus,id2word=dictionary, num_topics=10, passes=2, workers=4)

#%% LDA Topic Modeling
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))



#%%
#Doc2Vec
documents = [TaggedDocument(doc.split(' '), [i]) for i, doc in enumerate(df2["cleaned"])]
model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, workers=4)
model.build_vocab([x for x in tqdm(documents)])


for epoch in range(30):
    model.train(utils.shuffle([x for x in tqdm(documents)]), total_examples=len(documents), epochs=1)
    model.alpha -= 0.002
    model.min_alpha = model.alpha


doc2vec_df=pd.DataFrame()
for i in range(0,len(list(df2["tokens"]))):
    vector=pd.DataFrame(model.infer_vector(list(df2["tokens"])[i])).transpose()
    doc2vec_df=pd.concat([doc2vec_df,vector], axis=0)

doc2vec_df=doc2vec_df.reset_index(drop = True)


doc2vec_df_merged = df2.merge(doc2vec_df, left_index=True, right_index=True)


weight2 = doc2vec_df.to_numpy()

#doc2vec_df=doc2vec_df.drop('index', axis=1)


#%%
#TSNE with TF-IDF

tsne = TSNE(random_state=(99), perplexity = 100,learning_rate=200, n_jobs=(-1))
X = tsne.fit_transform(X)
X = StandardScaler().fit_transform(X)

#Graph TSNE to see if we can create clusters

x = X[:,0]
y = X[:,1]

fig, ax = plt.subplots()
ax.scatter(x,y)

ax.grid(True)
fig.tight_layout()

plt.show()

#%%
#TSNE with Doc2Vec
X2 = weight2
tsne = TSNE(random_state=(99), perplexity = 40,learning_rate=50, n_jobs=(-1))
X2 = tsne.fit_transform(X2)
X2 = StandardScaler().fit_transform(X2)

#Graph TSNE to see if we can create clusters

x = X2[:,0]
y = X2[:,1]

fig, ax = plt.subplots()
ax.scatter(x,y)

ax.grid(True)
fig.tight_layout()

plt.show()


#%%
#DBSCAN with TF_IDF

db = DBSCAN(eps=.15, min_samples=90).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

fig = plt.gcf()
fig.set_size_inches(35,18)

print("The Silhouette score is ", metrics.silhouette_score(X, labels, metric='sqeuclidean'))
for k, col in zip(unique_labels, colors):
    if k==-1:
        col = [0,0,0,1]
        
    class_member_mask = (labels == k)
    
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col),
             markersize=14)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col),
             markersize=6, )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.legend(labels = unique_labels, loc='upper right')
plt.show()

#pass back to df2
df2["_cluster_x"] = x
df2["cluster_y"] = y
df2["label"] = labels


#%%
#DBSCAN with Doc2Vec

db = DBSCAN(eps=.15, min_samples=90).fit(X2)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

fig = plt.gcf()
fig.set_size_inches(35,18)

print("The Silhouette score is ", metrics.silhouette_score(X2, labels, metric='sqeuclidean'))
for k, col in zip(unique_labels, colors):
    if k==-1:
        col = [0,0,0,1]
        
    class_member_mask = (labels == k)
    
    xy = X2[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col),
             markersize=14)
    
    xy = X2[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col),
             markersize=6, )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.legend(labels = unique_labels, loc='upper right')
plt.show()

#pass back to df2
df2["_cluster_x"] = x
df2["cluster_y"] = y
df2["label"] = labels

#%% K-Means
k=8
km = KMeans(n_clusters=k, random_state =89)
km.fit(TFIDF_matrix)
clusters = km.labels_.tolist()


terms = Tfidf.get_feature_names()
Dictionary={'Doc Name':str(df2.index + df2["variety"]), 'Cluster':clusters,  'Text': df2["cleaned"]}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name','Text'])

print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]


#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

#dictionary to store terms and titles
cluster_terms={}
cluster_title={}


for i in range(k):
    print("Cluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms
    
    print("Cluster %d titles:" % i, end='')
    temp=frame[frame['Cluster']==i]
    for title in temp['Doc Name']:
        print(' %s,' % title, end='')
        temp_titles.append(title)
    cluster_title[i]=temp_titles



#%%
#Topic Modeling- Tf-IDF

df2_label = df2[['label']]
df2_tfidf = df2_label.merge(matrix, right_index = True, left_index = True)

df2_topics_weights = df2_tfidf.groupby(by= "label").mean()


df2_topics = df2_topics_weights.apply(lambda x: pd.Series(x.sort_values(ascending=False)
       .iloc[:4].index, 
      index=['top1','top2','top3', 'top4']), axis=1).reset_index()

df2_topics["Topics"] = df2_topics[['top1','top2','top3', 'top4']].agg('-'.join, axis=1)
df2_topics = df2_topics[["label", "Topics"]]

plot = df.plot.pie(y='variety', figsize=(5, 5))

#%%
#Optuna for hyperparameter tuning

def optimize(trial, docs):
    #results
    silhouette_score = []
    
    #Vectorization
    #max_features = trial.suggest_int("max_features", 1000, 1500)
    min_df = trial.suggest_float("min_df", .05, .07)
    max_df = trial.suggest_float("max_df", .2, .3)
    
    Tfidf=TfidfVectorizer(ngram_range=(1,1), min_df=min_df, max_df = max_df)

    TFIDF_matrix=Tfidf.fit_transform(doc)

    matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names())


    weight = TFIDF_matrix.toarray()
    X = weight

    # t-SNE
    
    learning_rate = trial.suggest_int("learning_rate", 200, 300)
    perplexity = trial.suggest_float("perplexity", 20,100)
    
    
    tsne = TSNE(random_state=(99), perplexity = perplexity,learning_rate=learning_rate, n_jobs=(-1))
    X = tsne.fit_transform(X)
    X = StandardScaler().fit_transform(X)
    
    #cluster
    min_samples = trial.suggest_int("min_samples", 60, 100)
    eps = trial.suggest_float("eps", .1, .2)
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    labels = db.labels_
    
    #Metric
    score = metrics.silhouette_score(X, labels, metric = 'sqeuclidean')
    silhouette_score.append(score)
    
    return silhouette_score

optimization_function = partial(optimize, docs = doc)
study = optuna.create_study(direction=("maximize"))
study.optimize(optimization_function, n_trials = 10)


study.best_params


#%%
#Redo

doc = df2["cleaned"].tolist() #getting ready for TF-IDF

#TF-IDF
Tfidf=TfidfVectorizer(ngram_range=(1,1), min_df= study.best_params["min_df"], max_df = study.best_params["max_df"])

TFIDF_matrix=Tfidf.fit_transform(doc)

matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names())


weight = TFIDF_matrix.toarray()
X = weight

tsne = TSNE(random_state=(99), perplexity = study.best_params["perplexity"],learning_rate=study.best_params["learning_rate"], n_jobs=(-1))
X = tsne.fit_transform(X)
X = StandardScaler().fit_transform(X)


#Graph TSNE to see if we can create clusters

x = X[:,0]
y = X[:,1]

fig, ax = plt.subplots()
ax.scatter(x,y)

ax.grid(True)
fig.tight_layout()

plt.show()



#DBSCAN

db = DBSCAN(eps=study.best_params["eps"], min_samples=study.best_params["min_samples"]).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

fig = plt.gcf()
fig.set_size_inches(35,18)

print("The Silhouette score is ", metrics.silhouette_score(X, labels, metric='sqeuclidean'))
for k, col in zip(unique_labels, colors):
    if k==-1:
        col = [0,0,0,1]
        
    class_member_mask = (labels == k)
    
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col),
             markersize=14)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col),
             markersize=6, )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.legend(labels = unique_labels, loc='upper right')
plt.show()




#%%
#Modeling predicting points based on only on TF-IDF Reviews Matrix using Random Forest Regressor
y = df["points"]
X = matrix


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

#rf = RandomForestRegressor()
#rf.fit(X_train, y_train)

#Load pickled 

pickled_model = pickle.load(open('rf_nlp_model.pkl', 'rb'))
rf = pickled_model


y_pred = rf.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

rf.score(X_test, y_test)

pickle.dump(rf, open('rf_nlp_model.pkl', 'wb')) #save random-forest model


