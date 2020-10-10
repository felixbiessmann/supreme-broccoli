import re, string, os, glob, json
from random import random
import pandas as pd
from bs4 import BeautifulSoup
import requests
import GetOldTweets3 as got 
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from datetime import datetime
import itertools
from time import sleep

SAVEDIR = 'twitterdata'

START_BEFORE='2020-01-15'
STOP_BEFORE='2020-01-31'
START_AFTER='2020-04-01'
STOP_AFTER='2020-04-30'

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def get_text_from_url(url):
    if url:
        try:
            soup = BeautifulSoup(requests.get(url, timeout=0.1).text, 'html.parser')
            return " ".join([p.get_text().replace(u'\xa0', u' ') for p in soup.find_all('p')])
        except:
            return ""
    else:
        return ""

def get_tweets(keywords, start='2020-04-10', stop='2020-04-11',save_dir=''):
   # try:
    datestr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    savestr = os.path.join(save_dir,f'{keywords[0]}-{keywords[-1]}-{datestr}.json')
    print(f'{datestr}: Fetching tweets in range {start} - {stop} for keywords: {keywords}')

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(" OR ".join(keywords))\
                                         .setSince(start)\
                                         .setUntil(stop)\
                                         .setMaxTweets(0)\
                                         .setLang('de')
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    tweet_dicts = []
    for tweet in tweets:
        tweet_dict = tweet.__dict__ 
        tweet_dict['url_text'] = ""#get_text_from_url(tweet.urls)
        tweet_dicts.append(tweet_dict)

    enddatestr =  datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    print(f'{enddatestr}: Found {len(tweets)} tweets')
    if len(tweets)>0:
        pd.DataFrame(tweet_dicts).set_index('date').to_json(savestr, orient='records', lines=True)
    #except:
    #    print('Error getting tweets')
    #    pass

def get_tweets_for_keywords(batchsize=5, max_keywords=200):
    df = pd.read_csv('keywords.csv') 

    for manifestolabel in df.columns:
        path = os.path.join(SAVEDIR, manifestolabel)
        os.makedirs(path, exist_ok=True)
        for kw_chunk in chunked_iterable(df[manifestolabel].dropna()[:max_keywords], batchsize): 
            get_tweets(keywords=kw_chunk, 
                       start=START_BEFORE,
                       stop=STOP_BEFORE, 
                       save_dir=path)
            sleep(random() * 5 + 2)
            get_tweets(keywords=kw_chunk, 
                       start=START_AFTER,
                       stop=STOP_AFTER, 
                       save_dir=path)
            sleep(random() * 5 + 2)

def read_json_tweets():
    dfs = []
    files = glob.glob(os.path.join(SAVEDIR, "**", "*.json"), recursive=True)
    print(f'Found {len(files)} json files in {SAVEDIR}')
    for file in files:
        json_lines = open(os.path.join(file)).readlines()
        print(f'\tReading {len(json_lines)} lines from {file}')
        df = pd.DataFrame([json.loads(line) for line in json_lines])
        df['manifestolabel_keywords'] = file.split('/')[-2]
        df['date'] = pd.to_datetime(df['formatted_date']).dt.tz_localize(None)
        dfs.append(df)        
    return pd.concat(dfs).reset_index(drop=True)

def read_csv_tweets():
    files = glob.glob(os.path.join(SAVEDIR, "**", "*.csv"), recursive=True)
    print(f'Found {len(files)} json files in {SAVEDIR}')
    dfs = []
    for file in files:

        df = pd.read_csv(open(file,'rU'), encoding='utf8', engine='c' ,
                          error_bad_lines=False, 
                          parse_dates=True,
                          infer_datetime_format=True)
        print(f'\tRead {len(df)} lines from {file}')    
        df['manifestolabel_keywords'] = file.split('/')[-2]
        df['date'] = pd.to_datetime(df['formatted_date']).dt.tz_localize(None)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)

def concat_all_tweets():
    df = pd.concat([read_json_tweets(), read_csv_tweets()]).drop_duplicates(subset='id')

    df['before'] = df['date'] < pd.Timestamp(2020,3,15)
    df['after'] = df['date'] > pd.Timestamp(2020,3,15)

    df.to_csv('all_tweets.csv')
    df.to_pickle('all_tweets.pickle')
