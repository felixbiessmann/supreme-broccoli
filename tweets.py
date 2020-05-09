import re, string
import pandas as pd
from bs4 import BeautifulSoup
import requests
import GetOldTweets3 as got 
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from datetime import datetime
import itertools

SAVEDIR = 'twitterdata'

START_BEFORE='2020-02-01'
STOP_BEFORE='2020-02-14'
START_AFTER='2020-04-01'
STOP_AFTER='2020-04-14'

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
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            return " ".join([p.get_text().replace(u'\xa0', u' ') for p in soup.find_all('p')])
        except:
            return ""
    else:
        return ""

def get_tweets(keywords, start='2020-04-10', stop='2020-04-11',save_dir=''):
    datestr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    savestr = f'{keywords[0]}-{keywords[-1]}-{datestr}.csv'
    print(f'{datestr}: Fetching tweets for keywords: {keywords}')

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(" OR ".join(keywords))\
                                         .setSince(start)\
                                         .setUntil(stop)\
                                         .setMaxTweets(0)\
                                         .setLang('de')\
                                         .setNear('Berlin, Germany')\
                                         .setWithin('1000km') 
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    tweet_dicts = []
    for tweet in tweets:
        tweet_dict = tweet.__dict__ 
        tweet_dict['url_text'] = get_text_from_url(tweet.urls)
        tweet_dicts.append(tweet_dict)

    enddatestr =  datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    print(f'{enddatestr}: Found {len(tweets)} tweets for keywords: {keywords}')
    pd.DataFrame(tweet_dicts).set_index('date').to_csv(savestr, index=False)
    

def get_tweets_for_keywords(batchsize=20, max_keywords=40):
	df = pd.read_csv('keywords.csv') 
	for manifestolabel in df.columns:
		path = os.path.join(SAVEDIR, manifestolabel)
        os.makedirs(path)
        for kw_chunk in chunked_iterable(df[kw].dropna()[:max_keywords], batchsize): 
            get_tweets(keywords=kw_chunk, 
                       start=START_BEFORE,
                       stop=STOP_BEFORE, 
                       save_dir=path)
            get_tweets(keywords=kw_chunk, 
                       start=START_AFTER,
                       stop=STOP_AFTER, 
                       save_dir=path)
		