import re, string
import pandas as pd
from bs4 import BeautifulSoup
import requests
import GetOldTweets3 as got 
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from datetime import datetime

def get_tweets(keywords, start='2020-04-10', stop='2020-04-11'):
    datestr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    savestr = f'{keywords[0]}-{keywords[-1]}-{datestr}.csv'
    print(f'Fetching tweets for keywords: {keywords}')
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

    print(f'Found {len(tweets)} tweets for keywords: {keywords}')
    pd.DataFrame(tweet_dicts).set_index('date').to_csv(savestr, index=False)
    
