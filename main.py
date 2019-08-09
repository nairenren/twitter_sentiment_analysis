# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:02:07 2018

@author: REN
"""

import re
import tweepy
from nltk.classify import NaiveBayesClassifier
import numpy as npnp
import pandas as pd
from tweepy import OAuthHandler
from textblob import TextBlob
import matplotlib.pyplot as plt
#from IPython.display import display
#import seaborn as sns
#==============================================================================
def word_feats(words):
    #Mengubah kata menjadi dictionary huruf
    return dict([(word, True) for word in words])
#==============================================================================
#Inisialisasi data training
train = pd.read_csv('trainset.csv')

temp_pos = np.array(train.loc[:, 'positive'])
print temp_pos
temp_neg = np.array(train.loc[:, 'negative'])
temp_neu = np.array(train.loc[:, 'neutral'])
    
positive_vocab = [x for x in temp_pos if str(x) != 'nan']
neutral_vocab = [x for x in temp_neu if str(x) != 'nan']
negative_vocab = [x for x in temp_neg if str(x) != 'nan']

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]

#Train data dengan klasifikasi NaiveBayes
train_set = negative_features +  positive_features + neutral_features
classifier = NaiveBayesClassifier.train(train_set)
#==============================================================================
class TwitterClient(object): 
    def __init__(self):
        #Inisialisasi key dan token dari Twitter Dev Console
        consumer_key = '9xErNzK5jJ1ODKY9JOBBgsCbm'
        consumer_secret = '4NisQWtc15duAXlQ17C8tW8ZHot8zMfXL3eaYV6v68n6UsZHcL'
        access_token = '187390279-NA5ot8jEZJIbwHBvDLvR7qyuuc6ic1YxjppoG876'
        access_token_secret = 'iS3wwnDlpKU6rZhFwbm7kbNVhAHrrdT7ty8sQNNv57Ymd'
        try:
            #Membuat objek OAuthHandler
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
#==============================================================================
    def clean_tweet(self, tweet):
        #Return data tweet yang telah dibersihkan 
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet).split())
#============================================================================== 
    def get_tweet_sentiment(self, tweet):
        #Membuat objek TextBlob untuk setiap tweet
        analysis = TextBlob(self.clean_tweet(tweet))
        
        #Inisialisasi sentimen dan kalimat tweet
        pos = neg = neu = 0
        sentence = analysis.lower()
        words = sentence.split(' ')
        for word in words:
            #Klasifikasi sentimen setiap kata per-huruf dari tweet
            classResult = classifier.classify(word_feats(word))
            if classResult == 'pos':
                pos += 1
            elif classResult == 'neg':
                neg += 1
            elif classResult == 'neu':
                neu += 1
            #print(word, " ", classResult)
        #print(pos, " ",neg, " ",neu)
        
        #Return hasil klasifikasi sentimen
        if pos > neg and pos >= neu:
            return 1
        elif neg > pos and neg >= neu:
            return -1
        else:
            return 0
#============================================================================== 
    def get_tweets(self, query, count=150):
        #Inisialisasi list untuk menyimpan tweet
        tweets = []
 
        try:
            #Panggil API Twitter untuk stream data tweet
            fetched_tweets = self.api.search(q=query, count=count, \
                                             until='2018-11-05')
            print("Number of tweets extracted: {}.\n".format(len \
                  (fetched_tweets)))
 
            #Parse data tweet untuk setiap tweet yang diambil
            for tweet in fetched_tweets:
                #Inisialisasi dictionary untuk menyimpan parameter tweet
                parsed_tweet = {}
 
                #Menyimpan parameter dari tweet
                parsed_tweet['text'] = tweet.text
                parsed_tweet['RTs'] = tweet.retweet_count
                parsed_tweet['date'] = tweet.created_at
                parsed_tweet['source'] = tweet.source
 
                #Append tweet yang telah di-parse ke list tweets
                if tweet.retweet_count > 0:
                    #Memastikan tweet yang di-RT hanya di-append sekali
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            #Return tweet yang telah di-parse
            return tweets
        except tweepy.TweepError as e:
            #Print error (Jika ada)
            print("Error : " + str(e))
#==============================================================================
def main():
    #Buat objek class TwitterClient
    api = TwitterClient()
    
    #Tarik data tweets berdasarkan query
    tweets = api.get_tweets(query='Dilan 1990')
    print("Number of tweets filtered: {}.\n".format(len(tweets)))
    
    #Petakan data, panjang dan sentimen tweets ke dalam Dataframe pandas
    data = pd.DataFrame(data = [tweet['text'] \
                              for tweet in tweets], columns=['Tweets'])
    data['len']  = np.array([len(tweet['text']) for tweet in tweets])
    data['Retweet'] = np.array([tweet['RTs'] for tweet in tweets])
    data['Date'] = np.array([tweet['date'] for tweet in tweets])
    data['Source'] = np.array([tweet['source'] for tweet in tweets])
    data['SA'] = np.array([api.get_tweet_sentiment(tweet) \
        for tweet in data['Tweets']])
    
    #Tampilkan (data.head(10))
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 2):
        print(data)
#==============================================================================
    #Inisialisasi persentase sentimen pada plot 
    sources = [1, 0, -1]
    percent = np.zeros(len(sources))
    
    for m in data['SA']:
        for n in range(len(sources)):
            if m == sources[n]:
                percent[n] += 1
                pass
    percent = (percent * 100)/len(data['Tweets'])
    
    print("\nPositive tweets percentage: {0:.2f} %".format(percent[0]))
    print("Neutral tweets percentage: {0:.2f} %".format(percent[1]))
    print("Negative tweets percentage: {0:.2f} %".format(percent[2]))
    
    #Plotting data sentimen ke grafik pie
    chart = pd.Series(percent, index=['Positive', 'Neutral', 'Negative'], \
                      name='Sentiment (%)')
    chart.plot.pie(fontsize=12, colors=['#c2c2f0','#ffb3e6', '#6f6fdc'], \
                   autopct='%.2f', figsize=(6, 6), \
                   pctdistance=0.8, explode=(0, 0, 0))
    centre_circle = plt.Circle((0, 0), 0.60, fc='#ffffff')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.show()
#==============================================================================
    try:
        #Menyimpan data dan hasil analisis sentimen dari tweets
        data.to_csv(path_or_buf='sentiment_analysis.csv', sep=',', \
                    na_rep='null', float_format=None, index_label=True, \
                    mode='w', encoding='utf-8', compression=None, \
                    quoting=None, quotechar='"', line_terminator = '\n', \
                    chunksize=None, date_format=None, decimal='.')
    except:
        print("Error : Failed to write to .csv")
#==============================================================================
if __name__ == "__main__":
    #Panggil fungsi main
    main()