# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 18:50:06 2022

@author: Alex
"""

#%%
import re,string
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
import pickle
from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
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

from gensim.models import Word2Vec,LdaMulticore, TfidfModel
from gensim import corpora


from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import numpy as np

# machine learning
import tensorflow as tf
from tensorflow import keras

#%%
# Load file
currdir = os.getcwd()
os.chdir(r"C:\Users\Alex\Desktop\Northwestern MSDS\MSDS 453\Final Project")
# read data

df = pd.read_csv("winemag-data_first150k.csv", index_col=(0))

df.isna().sum()

#%% Cleaning Data

df.dropna(inplace = True, subset=("price", "region_1", "variety")) #dropping rows that does not have a price or region
df.drop_duplicates(subset=("description"), inplace= True)
df.reset_index(drop = True, inplace= True)

# Look at wine varieties
font={'family': 'normal','weight': 'normal', 'size': 26}
plt.rc('font',**font)

variety_df = df.groupby('variety').filter(lambda x: len(x) > 500) # drop
varieties = variety_df['variety'].value_counts().index.tolist()
fig, ax = plt.subplots(figsize = (25, 10))
sns.countplot(x = variety_df['variety'], order = varieties, ax = ax).set(title='Types of Wine by Frequency in Data')
plt.xticks(rotation=90)

# drop wine varieties with not enough reviews
df_variety = df[df['variety'].isin(varieties)]

col = df.columns

#%% clean doc

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

df_variety["tokens"] = df_variety["description"].apply(lambda x: clean_doc(x)) #create tokens
df_variety["cleaned"] = df_variety["tokens"].apply(lambda x: str(" ".join(x))) #stitching

#%% TFIDF

doc = df_variety["cleaned"].tolist()

#function
def tfidf_func(doc, ngram, features):
    tfidf = TfidfVectorizer(ngram_range=(1,ngram), max_features=features)
    TFIDF_matrix = tfidf.fit_transform(doc)
    df_tfidf=pd.DataFrame(TFIDF_matrix.toarray(), columns = tfidf.get_feature_names())
    return df_tfidf


#TF-IDF
Tfidf=TfidfVectorizer(ngram_range=(1,3), max_features=300)

TFIDF_matrix=Tfidf.fit_transform(doc)

df_tfidf=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names())


weight = TFIDF_matrix.toarray()
X = weight

#%% Setup for Machine Learning Methods

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

# encode categorical classes
target_var = df_variety['variety']
label_encoder = LabelEncoder()
target = np.array(label_encoder.fit_transform(target_var))


# split into train test split
X_train, X_test, y_train, y_test = train_test_split(df_tfidf, target, test_size=0.2, random_state=42)

def plot_confusion_matrix(y_true,y_pred):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(5,5))
    matrix = confusion_matrix(y_true, y_pred)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    #plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    # plt.savefig("confusion_matrix_plot_mnist", tight_layout=False)
    plt.show()


#%% Random Forest Classifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest ClassifierAccuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

# cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
n_scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#%% Gradient Boosting Classifier

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Gradient Boosting Classifier Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

# cross validation
# cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
# n_scores = cross_val_score(gb, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


#%% XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Classifier Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred)

# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
# n_scores = cross_val_score(xgb, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#%% Neural Network Model

def plot_confusion_matrix(y_true,y_pred):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(5,5))
    matrix = confusion_matrix(y_true, y_pred)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    #plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    # plt.savefig("confusion_matrix_plot_mnist", tight_layout=False)
    plt.show()


model1 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[500]),
    keras.layers.Dense(250, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(28, activation="softmax")])

model1.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

model1.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

evaluation1 = model1.evaluate(X_test, y_test)

y_pred = model1.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Neural Network Accuracy: %.2f%%" % (accuracy * 100.0))
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred)
print("All models run")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
n_scores = cross_val_score(model1, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


#%% Find best performance

ngram_iter = [3]
features_iter = [50,100,300,500]
performance_acc = []

for i in ngram_iter:
    for j in features_iter:
        # create tfidf vector
        df_tfidf = tfidf_func(doc,i,j)
        
        # perform model
        xgb = XGBClassifier()
        
        # encode categorical classes
        target_var = df_variety['variety']
        label_encoder = LabelEncoder()
        target = np.array(label_encoder.fit_transform(target_var))


        # split into train test split
        X_train, X_test, y_train, y_test = train_test_split(df_tfidf, target, test_size=0.2, random_state=42)

        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)

        # evaluate predictions
        accuracy = accuracy_score(y_test, y_pred)
        performance_acc.append(accuracy)
        # print(f'For {ngram_iter[i]} n-grams and {features_iter[j]} max feature size the accuracy was: {accuracy} ')

performance_acc

#%% Plot F1 Score for countries

f1_scores = [0.90, 0.82, 0.77, 0.48, 0.40, 0.47]
f1_countries = ['United States', 'Italy', 'France', 'Spain', 'Argentina', 'Australia']

from matplotlib.pyplot import figure

figure(figsize=(10,5))

plt.plot(f1_countries,f1_scores)
plt.title('F-Score Variation by Country')
plt.xlabel('Country')
plt.ylabel('F-Score')
plt.grid()
plt.rcParams.update({'font.size': 20})
plt.show

#%% Plot f scores for types of wine

f1_scores_type = [0.56, 0.49, 0.62, 0.35, 0.73, 0.39, 0.39, 0.47, 0.54, 0.42, 0.53, 0.39, 0.26, 0.71, 0.55, 0.49, 0.65, 0.48, 0.39, 0.27, 0.58, 0.61, 0.35, 0.58, 0.35, 0.55, 0.52, 0.49]
f1_type_support = [581, 180, 1451, 128, 1544, 161, 111, 363, 543, 195, 106, 158, 129, 1661, 1028, 204, 295, 315, 389, 124, 472, 213, 235, 636, 338, 154, 273, 438]
f1_type_df = pd.DataFrame({'f1_scores':f1_scores_type,'support':f1_type_support})
sorted_df = f1_type_df.sort_values('support',axis=0, ascending=False)



sorted_df['type'] = varieties
f1_x = sorted_df['type'].tolist()
f1_y = sorted_df['f1_scores'].tolist()

from matplotlib.pyplot import figure

figure(figsize=(30,10))

plt.plot(f1_x, f1_y)
plt.title('F-Score Variation by Type of Wine')
plt.xlabel('Type of Wine')
plt.ylabel('F-Score')
plt.grid()
plt.rcParams.update({'font.size': 20})
plt.xticks(rotation=90)
plt.show()

