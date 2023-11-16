# -*- coding: utf-8 -*-
import datetime

import pandas
# Goto Anaconda Prompt and start as administrator
# conda install -c anaconda pymysql
from sqlalchemy import create_engine
from sqlalchemy import text

# load data from database
engine = create_engine('mysql+pymysql://hochstein:hochstein@bwa2.f4.htw-berlin.de:3306/reuters?charset=utf8')
conn = engine.connect()
stmt = text("SELECT * FROM reuters_news")
res = conn.execute(stmt)
data = pandas.DataFrame(res.fetchall())
data.columns = res.keys()
res.close()

# drop records with na values
data = data.dropna()
data = data.reset_index(drop=True)
data['Date'] = pandas.to_datetime(data['Date'])

# extract features from text
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(min_df=200)  # token_pattern = r'(?u)\b[A-Za-z]+\b'
X = count_vect.fit_transform(data['News'])  # document_term_matrix or bag_of_words
dtm = pandas.DataFrame(X.toarray())
dtm.columns = count_vect.get_feature_names()
data_dtm = pandas.concat([data, dtm], axis=1)

# handling dates
data_dtm['DateOnly'] = pandas.to_datetime(data_dtm['Date'].apply(datetime.datetime.date))

# import stock price data
prices = pandas.read_csv('prices.csv')

# handling dates
prices['Date'] = pandas.to_datetime(prices['Date'], infer_datetime_format=True)

# get prices day after event
prices['Price_Next_Day'] = prices['Close'].groupby(prices['Symbol']).shift(periods=-1)
# get prices for 5 days hold
prices['Price_Hold_5_Days'] = prices['Close'].groupby(prices['Symbol']).shift(periods=-6)
# calculate return as percentage
prices['Return_Hold_5_Days'] = (pandas.to_numeric(prices['Price_Hold_5_Days']) - pandas.to_numeric(
    prices['Price_Next_Day'])) / pandas.to_numeric(prices['Price_Next_Day'])

# merge datasets
data_processed = pandas.merge(data_dtm, prices[['Return_Hold_5_Days', 'Symbol', 'Date']], how='left',
                              left_on=['StockID', 'DateOnly'], right_on=['Symbol', 'Date'])
data_processed = data_processed.dropna()
data_processed = data_processed.reset_index(drop=True)

# transform into binary classification problem (no dealing with outliers, normalization/scaling issues etc.)
data_processed['Return_Boolean'] = data_processed['Return_Hold_5_Days'].apply(lambda x: 'pos' if x > 0.005 else 'neg')

# generate training and test data
import random

sample = random.sample(range(len(data_processed.index)), k=int(len(data_processed.index) * 0.8))
training = data_processed.iloc[sample]
test = data_processed.drop(sample)

# apply classifier
# http://scikit-learn.org/stable/tutorial/machine_learning_map/
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
forest.fit(training.iloc[:, 5:-5], training['Return_Boolean'])
forest.score(test.iloc[:, 5:-5], test['Return_Boolean'])

# explore predictions
pred_class = forest.predict(test.iloc[:, 5:-5])
test_copy = test.copy()
test_copy['pred_class'] = pred_class
probabilities = forest.predict_proba(test.iloc[:, 5:-5])
test_copy['probabilities'] = probabilities[:, 1]

# consider problem of inbalanced datasets, e.g. spam filter
from sklearn.metrics import confusion_matrix

cfm = confusion_matrix(test['Return_Boolean'], pred_class, labels=['pos', 'neg'])

# explore feature weights
importance_df = pandas.DataFrame(forest.feature_importances_)
importance_df['feature'] = test.columns[5:-5]
