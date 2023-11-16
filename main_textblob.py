from textblob import TextBlob
import pandas as pd

from sklearn.utils import shuffle

import time

import json


import dataProcesser as dp

# target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
df = pd.read_csv('data/twitter/kaggle_twitter.csv', encoding='iso-8859-1')
df_tt = df.rename(columns={'clean_text': 'text', 'category': 'target'})
df_tt.dropna()
df_tt = shuffle(df_tt, random_state=42)
df_tt = df_tt.iloc[20000:25000]


# train, test = train_test_split(df_tt, test_size=0.2, random_state=42)

# expects a dataframe with columns 'text' and 'target'
def evaluate_with_textblob_3(df_tb):
    correct_assumptions = 0
    accuracy = 0
    start_time = time.time()
    i = 0

    for index, row in df_tb.iterrows():
        i += 1
        testimonial = TextBlob(row['text'])
        sentiment = 0

        if testimonial.sentiment.polarity < -0.33:
            sentiment = -1
        elif testimonial.sentiment.polarity > 0.33:
            sentiment = 1
        else:
            sentiment = 0

        if sentiment == row['target']:
            accuracy += 1

        if i % 20000 == 0 or i % df_tb.shape[0] == 0:
            elapsed = time.time() - start_time
            print('| {:7d}/{:7d} batches | time: {:5.2f}s |'.format(i, df_tb.shape[0], elapsed))
            start_time = time.time()

    accuracy = accuracy / df_tb.shape[0] * 100
    print(str(accuracy) + '%')


# expects a dataframe with columns 'text' and 'target'
def evaluate_with_textblob_5(df_tb):
    correct_assumptions = 0
    accuracy = 0
    start_time = time.time()
    i = 0

    for index, row in df_tb.iterrows():
        i += 1
        testimonial = TextBlob(row['text'])
        sentiment = -1

        if 0.6 < testimonial.sentiment.polarity <= 1.0:
            sentiment = 4
        elif 0.2 < testimonial.sentiment.polarity <= 0.6:
            sentiment = 3
        elif -0.2 <= testimonial.sentiment.polarity <= 0.2:
            sentiment = 2
        elif -0.6 <= testimonial.sentiment.polarity < -0.2:
            sentiment = 1
        elif -1 <= testimonial.sentiment.polarity < -0.6: 

            sentiment = 0

        if sentiment == row['target']:
            accuracy += 1

        if i % 20000 == 0 or i % df_tb.shape[0] == 0:
            elapsed = time.time() - start_time
            print('| {:7d}/{:7d} batches | time: {:5.2f}s |'.format(i, df_tb.shape[0], elapsed))
            start_time = time.time()

    accuracy = accuracy / df_tb.shape[0] * 100
    print(str(accuracy) + '%')


train, test, dev = dp.load_movie_reviews2()
train = train.rename(columns={"sentence": "text", "label": "target"})
test = test.rename(columns={"sentence": "text", "label": "target"})
dev = dev.rename(columns={"sentence": "text", "label": "target"})

evaluate_with_textblob_3(df_tt)
evaluate_with_textblob_5(train)
evaluate_with_textblob_5(test)
evaluate_with_textblob_5(dev)
