# Source: https://towardsdatascience.com/building-sentiment-classifier-using-spacy-3-0-transformers-c744bfc767b
# Model: https://spacy.io/models/en#en_core_web_trf

# Importing libraries

import time
from datetime import datetime

import pandas as pd
import spacy
from sklearn.utils import shuffle
from spacy.tokens import DocBin

import dataProcesser as dp

# Storing docs in binary format
spacy.prefer_gpu()

tw_train, tw_test, dev = dp.load_movie_reviews2()
tw_train = tw_train.rename(columns={"sentence": "text", "label": "target"})
tw_test = tw_test.rename(columns={"sentence": "text", "label": "target"})
dev = dev.rename(columns={"sentence": "text", "label": "target"})

tw_df = pd.read_csv('data/twitter/kaggle_twitter.csv', encoding='iso-8859-1')
tw_df = tw_df.rename(columns={'clean_text': 'Text', 'category': 'Sentiment'})
tw_df = tw_df.dropna()
tw_df = shuffle(tw_df, random_state=42)
tw_eval = tw_df.iloc[20000:25000]
tw_df = tw_df.head(20000)


def convert_number_to_label_twitter(number):
    if number == -1:
        label = "negative"
    elif number == 0:
        label = "neutral"
    else:
        label = "positive"
    return label


tw_df['Sentiment'] = tw_df['Sentiment'].apply(convert_number_to_label_twitter)
tw_eval['Sentiment'] = tw_eval['Sentiment'].apply(convert_number_to_label_twitter)

tw_train = tw_df.sample(frac=0.8, random_state=25)
tw_test = tw_df.drop(tw_train.index)

mo_train, mo_test, mo_eval = dp.load_movie_reviews2()
mo_train = mo_train.rename(columns={'sentence': 'Text', 'label': 'Sentiment'})
mo_test = mo_test.rename(columns={'sentence': 'Text', 'label': 'Sentiment'})
mo_eval = mo_eval.rename(columns={'sentence': 'Text', 'label': 'Sentiment'})

def convert_number_to_label_movies(number):
    if number == 0:
        label = "very negative"
    elif number == 1:
        label = "negative"
    elif number == 2:
        label = "neutral"
    elif number == 3:
        label = "positive"
    else:
        label = "very positive"
    return label

mo_train['Sentiment'] = mo_train['Sentiment'].apply(convert_number_to_label_movies)
mo_test['Sentiment'] = mo_test['Sentiment'].apply(convert_number_to_label_movies)
mo_eval['Sentiment'] = mo_eval['Sentiment'].apply(convert_number_to_label_movies)

nlp = spacy.load("en_core_web_trf")

# Creating tuples
tw_train['tuples'] = tw_train.apply(lambda row: (row['Text'], row['Sentiment']), axis=1)
tw_train = tw_train['tuples'].tolist()
tw_test['tuples'] = tw_test.apply(lambda row: (row['Text'], row['Sentiment']), axis=1)
tw_test = tw_test['tuples'].tolist()

mo_train['tuples'] = mo_train.apply(lambda row: (row['Text'], row['Sentiment']), axis=1)
mo_train = mo_train['tuples'].tolist()
mo_test['tuples'] = mo_test.apply(lambda row: (row['Text'], row['Sentiment']), axis=1)
mo_test = mo_test['tuples'].tolist()

# User function for converting the train and test dataset into spaCy document
def document_twitter(data):
    # Creating empty list called "text"
    text = []
    for doc, label in nlp.pipe(data, as_tuples=True):
        if (label == 'positive'):
            doc.cats['positive'] = 1
            doc.cats['negative'] = 0
            doc.cats['neutral'] = 0
        elif (label == 'negative'):
            doc.cats['positive'] = 0
            doc.cats['negative'] = 1
            doc.cats['neutral'] = 0
        else:
            doc.cats['positive'] = 0
            doc.cats['negative'] = 0
            doc.cats['neutral'] = 1

        # Adding the doc into the list 'text'
        text.append(doc)
    return (text)

def document_movies(data):
    # Creating empty list called "text"
    text = []
    for doc, label in nlp.pipe(data, as_tuples=True):
        if (label == 'very positive'):
            doc.cats['very positive'] = 1
            doc.cats['positive'] = 0
            doc.cats['neutral'] = 0
            doc.cats['negative'] = 0
            doc.cats['very negative'] = 0
        elif (label == 'positive'):
            doc.cats['very positive'] = 0
            doc.cats['positive'] = 1
            doc.cats['neutral'] = 0
            doc.cats['negative'] = 0
            doc.cats['very negative'] = 0
        elif (label == 'neutral'):
            doc.cats['very positive'] = 0
            doc.cats['positive'] = 0
            doc.cats['neutral'] = 1
            doc.cats['negative'] = 0
            doc.cats['very negative'] = 0
        elif (label == 'negative'):
            doc.cats['very positive'] = 0
            doc.cats['positive'] = 0
            doc.cats['neutral'] = 0
            doc.cats['negative'] = 1
            doc.cats['very negative'] = 0
        else:
            doc.cats['very positive'] = 0
            doc.cats['positive'] = 0
            doc.cats['neutral'] = 0
            doc.cats['negative'] = 0
            doc.cats['very negative'] = 1

        # Adding the doc into the list 'text'
        text.append(doc)
    return (text)

def convert_to_binary(train, test):
    # Calculate the time for converting into binary document for train dataset

    start_time = datetime.now()

    # passing the train dataset into function 'document'
    train_docs = document_twitter(train)

    # Creating binary document using DocBin function in spaCy
    doc_bin = DocBin(docs=train_docs)

    # Saving the binary document as train.spacy
    doc_bin.to_disk("more/train.spacy")
    end_time = datetime.now()

    # Printing the time duration for train dataset
    print('Duration: {}'.format(end_time - start_time))

    # Calculate the time for converting into binary document for test dataset

    start_time = datetime.now()

    # passing the test dataset into function 'document'
    test_docs = document_twitter(test)
    doc_bin = DocBin(docs=test_docs)
    doc_bin.to_disk("more/valid.spacy")
    end_time = datetime.now()

    # Printing the time duration for test dataset
    print('Duration: {}'.format(end_time - start_time))

def evaluate(df_eval, dataset):
    print('Evaluating spaCy model trained on '+dataset+' dataset...')
    nlp = spacy.load("output_updated_"+dataset+"/model-best")
    correct = 0
    count = 0
    start_time = time.time()
    for index, row in df_eval.iterrows():
        text = row['Text']
        label = row['Sentiment']
        demo = nlp(text)
        pred_label = max(demo.cats, key=demo.cats.get)
        if label == pred_label:
            correct += 1
        count += 1

        if count % 200 == 0 or count == df_eval.shape[0]:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} rows | time: {:5.2f}s | overall accuracy: {:2.2f}%'.format(count, df_eval.shape[0], elapsed,
                                                                                            correct / count * 100))
            start_time = time.time()