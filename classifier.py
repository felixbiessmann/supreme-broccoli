# -*- coding: utf-8 -*-
import pickle, os, gzip
import urllib.request
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from manifesto_data import (get_manifesto_texts,
                            DATADIR,
                            LABEL2DOMAIN,
                            LABEL2RIGHTLEFT,
                            MANIFESTOCODE2LABEL)

def get_bundestag_data(DATADIR='bundestagsprotokolle'):
    if not os.path.exists(DATADIR): 
        os.mkdir(DATADIR)

    file_name = os.path.join(DATADIR, 'bundestags_parlamentsprotokolle.csv.gzip')
    if not os.path.exists(file_name):
        url_data = 'https://www.dropbox.com/s/1nlbfehnrwwa2zj/bundestags_parlamentsprotokolle.csv.gzip?dl=1'
        urllib.request.urlretrieve(url_data, file_name)

    df = pd.read_csv(gzip.open(file_name), index_col=0).sample(frac=1)
    df.loc[df.wahlperiode==17,'government'] = df[df.wahlperiode==17].partei.isin(['cducsu','fdp'])
    df.loc[df.wahlperiode==18,'government'] = df[df.wahlperiode==18].partei.isin(['cducsu','spd'])
    
    return df

def get_manifesto_data():
    data,labels = get_manifesto_texts()
    df = pd.DataFrame({"text":data, "manifestocodes":labels})
    df['domain'] = df['manifestocodes']\
                    .apply(lambda x: LABEL2DOMAIN.get(x // 100, None))
    df['rightleft'] = df['manifestocodes']\
                    .apply(lambda x: LABEL2RIGHTLEFT.get(x, None))
    df['manifestolabel'] = df['manifestocodes']\
                    .apply(lambda x: MANIFESTOCODE2LABEL.get(x, None))\
                    .replace(['undefined'], [None])
    return df


def train_single(data, labels, save_str=""):
    '''

    Trains a classifier on bag of word vectors

    INPUT
    folds   number of cross-validation folds for model selection

    '''
    text_clf = Pipeline([('vect', TfidfVectorizer()),
                        ('clf',SGDClassifier(loss="log"))])
    parameters = {'vect__ngram_range': [(1,1)],
           'clf__alpha': (10.**np.arange(-6,-3)).tolist()}
    # perform gridsearch to get the best regularizer
    gs_clf = GridSearchCV(text_clf, parameters, cv=2, n_jobs=-1,verbose=6)
    gs_clf.fit(data,labels)
    # dump classifier to pickle
    fn = os.path.join(DATADIR, 'classifier-{}.pickle'.format(save_str))
    pickle.dump(gs_clf.best_estimator_,open(fn,'wb'))

def train_all(label_types = ['domain', 'rightleft', 'manifestolabel']):
    df = get_manifesto_data()
    for label_type in label_types:
        idx = df[label_type].isnull() == False
        train_single(df.loc[idx,'text'],df.loc[idx,label_type], label_type)

def score_texts(df, label_types = ['domain', 'rightleft', 'manifestolabel']):
    for label_type in label_types:
        fn = os.path.join(DATADIR, 'classifier-{}.pickle'.format(label_type))
        clf = pickle.load(open(fn,'rb'))
        df[label_type + "_proba"] = clf.predict_proba(df['text']).max(axis=1)
        df[label_type] = clf.predict(df['text'])
    return df

def get_keywords(label='manifestolabel', top_what=100):
    df = get_bundestag_data()
    fn = os.path.join(DATADIR, 'classifier-{}.pickle'.format(label))
    clf = pickle.load(open(fn,'rb'))
    labels_normalized = StandardScaler().fit_transform(clf.predict_proba(df['text']))
    vectorizer = clf.steps[0][1]
    data_scaled = StandardScaler(with_mean=False).fit_transform(vectorizer.transform(df['text']))
    keywords = {}
    for iclass, classname in enumerate(clf.steps[1][1].classes_):
        pattern = labels_normalized[:,iclass].T @ data_scaled
        idx2word = {idx: word for word, idx in vectorizer.vocabulary_.items()}
        keywords[classname] = [idx2word[idx] for idx in pattern.argsort()[-top_what:][::-1]]
    pd.DataFrame(keywords).to_csv('keywords.csv')
