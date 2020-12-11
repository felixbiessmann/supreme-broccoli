import re, string, os, glob, json
from random import random
import pandas as pd
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from datetime import datetime
import itertools
import snscrape.modules.twitter as sntwitter
from time import sleep
from multiprocessing import Pool, Process, Manager

SAVEDIR = 'twitterdata'
SAVEDIR_NEW = 'tweets'

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

def get_tweets(keywords, save_dir=SAVEDIR_NEW, maxTweets = 100):
    df_keywords = pd.read_csv('keywords.csv')
    
    dates = pd.date_range('1/1/2020', periods=52, freq='W')

    for week_idx, date in enumerate(dates):
        for label in df_keywords.columns:
            keywords = df_keywords[label]
            ss = label.replace('/','-')
            savestr = os.path.join(save_dir,f'{ss}-{date}.json')
            start = f'{dates[week_idx]}'[:10]
            stop = f'{dates[week_idx+1]}'[:10]
            print(f'{datestr}: Fetching tweets in range {start} - {stop} for keywords: {keywords}')

            tweets = []
            query = " OR ".join(keywords) + " lang:de" + ' since:' + start + " until:" + stop
            # Using TwitterSearchScraper to scrape data and append tweets to list
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i>maxTweets:
                    break
                tweets.append(tweet.__dict__)

            print(f'Found {len(tweets)} tweets')
            if len(tweets)>0:
                pd.DataFrame(tweets).to_json(savestr, orient='records', lines=True)


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

def read_new_tweet_json(fn, interaction_filter=True):
    df = pd.read_json(fn, orient='records', lines=True)
    print(f'\tRead {len(df)} lines from {fn}')    
    df = df[['url', 'date', 'content', 'id',  
            'replyCount', 'retweetCount', 'likeCount', 'quoteCount']]
    df.rename(columns = {'content':'text',
                                    'replyCount':'replies',
                                    'retweetCount':'retweets',
                                    'likeCount':'likes',
                                    'quoteCount':'quotes'
                                   }, inplace = True)
    df['manifestolabel_keywords'] = fn.split('/')[-1].split('-2020')[0]
    if interaction_filter:
        df = df[df[['replies','retweets','likes']].sum(axis=1)>1]
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    print(f'Found {len(df)} tweets')
    return df
    
def read_new_json_tweets(interaction_filter=False):
    dfs = []
    files = glob.glob(os.path.join(SAVEDIR_NEW, "**", "*.json"), recursive=True)
    print(f'Found {len(files)} json files in {SAVEDIR}')
    for file in files:
        df = read_new_tweet_json(file, interaction_filter)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        dfs.append(df)        
    return pd.concat(dfs).reset_index(drop=True)

def read_json_tweets(interaction_filter=False):
    dfs = []
    files = glob.glob(os.path.join(SAVEDIR, "**", "*.json"), recursive=True)
    print(f'Found {len(files)} json files in {SAVEDIR}')
    for file in files:
        json_lines = open(os.path.join(file)).readlines()
        print(f'\tReading {len(json_lines)} lines from {file}')
        df = pd.DataFrame([json.loads(line) for line in json_lines])
        df['manifestolabel_keywords'] = file.split('/')[-2]
        df['date'] = pd.to_datetime(df['formatted_date']).dt.tz_localize(None)
        df['text'] = df['text'].fillna('')
        df = df[['text','permalink','retweets','favorites','replies','date','manifestolabel_keywords']]\
            .rename(columns={'permalink':'url','favorites':'likes'})
        if interaction_filter:
            df = df[df[['replies','retweets','likes']].sum(axis=1)>1]
        print(f'Found {len(df)} tweets')
        dfs.append(df)        
    return pd.concat(dfs).reset_index(drop=True)

def read_csv_tweets(interaction_filter=False):
    
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
        df['text'] = df['text'].fillna('')
        df = df[['text','permalink','retweets','favorites','replies','date','manifestolabel_keywords']]\
            .rename(columns={'permalink':'url','favorites':'likes'})
        if interaction_filter:
            df = df[df[['replies','retweets','likes']].sum(axis=1)>1]
        print(f'Found {len(df)} tweets')
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)

def concat_all_tweets():
    df = pd.concat([read_json_tweets(), 
                    read_csv_tweets(), 
                    read_new_json_tweets()]).drop_duplicates(subset='id')

    df['before'] = df['date'] < pd.Timestamp(2020,3,15)
    df['after'] = df['date'] > pd.Timestamp(2020,3,15)

    df.to_csv('all_tweets.csv')
    df.to_pickle('all_tweets.pickle')