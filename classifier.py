# -*- coding: utf-8 -*-
import pickle, os, gzip
import requests, glob, json
import urllib.request
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from manifesto_data import (get_manifesto_texts,
                            DATADIR,
                            LABEL2DOMAIN,
                            LABEL2RIGHTLEFT,
                            MANIFESTOCODE2LABEL)

def load_bundestag_protocols():
    root_url = "https://www.bundestag.de"
    url = "https://www.bundestag.de/ajax/filterlist/de/services/opendata/543410-543410/h_49f0d94cb26682ff1e9428b6de471a5b?limit=10&noFilterSet=true&offset="
     
    for offset in range(0,300,10):
        protocol_list = requests.get(url + f"{offset}").text
     
        # Passing the source code to BeautifulSoup to create a BeautifulSoup object for it.
        soup = BeautifulSoup(protocol_list, 'lxml')

        # Extracting all the <a> tags into a list.
        tags = soup.find_all('a')
        os.makedirs('bundestagsprotokolle', exist_ok=True)

        for link in soup.find_all('a'):
            if link.get('href').endswith('.xml'):
                xml_response = requests.get(root_url + link.get('href'))
                open(os.path.join('bundestagsprotokolle', f'{offset}-' + link.get('href').split('/')[-1]), 'w').write(xml_response.text)

    files = glob.glob('bundestagsprotokolle/*data.xml')
    reden = []
    for file in files:
        protocol = BeautifulSoup(open(file).read())
        datum = protocol.find('veranstaltungsdaten').find('datum').get_text()
        for rede in protocol.find_all('rede'):
            reden.append({
                'datum': datum,
                'redner': rede.find('redner').get_text(),
                'rede': " ".join([x.getText() for x in rede.find_all('p',{'klasse':'J'})]),
                'beifall': [k.getText() for k in rede.find_all('kommentar') if 'Beifall' in k.getText()]
            })

    with open('alle_reden.json', 'w') as fp:
        json.dump(reden, fp)

def get_bundestag_data(DATADIR='bundestagsprotokolle'):

    df = pd.read_json(os.path.join(DATADIR, 'alle_reden.json')).sample(frac=1)
    
    return df

def get_manifesto_data():
    data,labels = get_manifesto_texts()
    df = pd.DataFrame({"text":data, "manifestocodes":labels})
    df['domain'] = df['manifestocodes']\
                    .apply(lambda x: LABEL2DOMAIN.get(x // 100, None))
    df['rightleft'] = df['manifestocodes']\
                    .apply(lambda x: LABEL2RIGHTLEFT.get(x, None))
    df['manifestolabel'] = df['manifestocodes']\
                    .apply(lambda x: MANIFESTOCODE2LABEL.get(x, None))
    return df

def get_human_data():
    df_human = pd.read_json('human_labeled_data.json')
    df_human['text'] = df_human['text_data'].str.replace('https[^\s]*\s', '')
    df_human['manifestolabel'] = df_human['major_label'].str.replace('\d\d\d ','')
    return df_human


def train_single(data, labels, save_str=""):
    '''

    Trains a classifier on bag of word vectors

    INPUT
    folds   number of cross-validation folds for model selection

    '''
    text_clf = Pipeline([('vect', TfidfVectorizer()),
                        ('clf',SGDClassifier(loss="log",penalty='l1',max_iter=2000,early_stopping=True))])
    parameters = {'vect__ngram_range': [(1,1), (1,2), (1,3)]}
    # perform gridsearch to get the best regularizer
    gs_clf = GridSearchCV(text_clf, parameters, cv=2, n_jobs=-1,verbose=6)
    gs_clf.fit(data,labels)
    # dump classifier to pickle
    fn = os.path.join(DATADIR, 'classifier-{}.pickle'.format(save_str))
    pickle.dump(gs_clf.best_estimator_,open(fn,'wb'))
    # 2-fold CV with best hps to evaluate and calculate precision thresholds
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2)
    eval_clf = gs_clf.best_estimator_.fit(X_train, y_train)
    y_hat = eval_clf.predict(X_test)
    metrics_df = pd.DataFrame(classification_report(y_test, y_hat, output_dict=True))
    fn_metrics = os.path.join(DATADIR, 'classifier-{}-metrics.csv'.format(save_str))
    metrics_df.to_csv(fn_metrics)

    y_hat_proba = eval_clf.predict_proba(X_test)
    prec_rec_curves = {}

    for label in eval_clf.classes_:
        prec, rec, thresh = precision_recall_curve(y_test==label, y_hat_proba[:,eval_clf.classes_==label])
        prec_rec_curves[label] = {
                                    'precisions':prec,
                                    'recall':rec,
                                    'thresholds':thresh
                                }
    fn_pr_curve = os.path.join(DATADIR, 'classifier-{}-pr_curve.pickle'.format(save_str))
    pickle.dump(prec_rec_curves, open(fn_pr_curve,'wb'))

def train_all(label_types = ['domain', 'rightleft', 'manifestolabel']):
    df = get_manifesto_data()
    for label_type in label_types:
        idx = df[label_type].isnull() == False
        train_single(df.loc[idx,'text'],df.loc[idx,label_type], label_type)

def score_texts(df, 
                label_types = ['domain', 'rightleft', 'manifestolabel'], 
                min_precision=0.5):
    for label_type in label_types:
        fn = os.path.join(DATADIR, 'classifier-{}.pickle'.format(label_type))
        clf = pickle.load(open(fn,'rb'))
        # first predict labels and their likelihood
        df[label_type + "_proba"] = clf.predict_proba(df['text']).max(axis=1)
        df[label_type] = clf.predict(df['text'])
        # now filter out predictions below specified precision threshold
        fn = os.path.join(DATADIR, 'classifier-{}-pr_curve.pickle'.format(label_type))
        pr_curves = pickle.load(open(fn,'rb'))
        for label in clf.classes_:
            above_precision = pr_curves[label]['thresholds'] > min_precision
            if sum(above_precision) > 0:
                prec_thresh = pr_curves[label]['thresholds'][above_precision.nonzero()[0][0]-1]
                below_precision_idx = (df[label_type]==label) & (df[label_type + "_proba"] < prec_thresh)
                df.loc[below_precision_idx, label_type] = ''
            else:
                df.loc[df[label_type]==label, label_type] = ''

    return df

def get_keywords(label='all_tweets', top_what=100):
    stopwords = [x.strip() for x in open('stopwords.txt').readlines()[7:]]

    df_bundestag = get_bundestag_data()
    df_manifesto = get_manifesto_data()
    df_human = get_human_data()
    text = pd.concat([df_bundestag['rede'],df_manifesto['text'],df_human['text']],axis=0)
    fn = os.path.join(DATADIR, 'classifier-{}.pickle'.format(label))
    clf = pickle.load(open(fn,'rb'))
    labels_normalized = StandardScaler().fit_transform(clf.predict_proba(text))
    vectorizer = clf.steps[0][1]
    data_scaled = StandardScaler(with_mean=False).fit_transform(vectorizer.transform(text))
    keywords = {}
    for iclass, classname in enumerate(clf.steps[1][1].classes_):
        if classname not in ['undefined', 'ignored']:
            pattern = labels_normalized[:,iclass].T @ data_scaled
            idx2word = {idx: word for word, idx in vectorizer.vocabulary_.items()}
            words = pd.Series([idx2word[idx] for idx in pattern.argsort()[::-1]])
            exclude = words.isin(stopwords) | words.str.contains('\d+')
            keywords[classname] = words[~exclude].dropna()[:top_what].values
    df_keywords = pd.DataFrame(keywords)
    df_keywords.to_csv('keywords.csv', index=False)
