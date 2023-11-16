import gzip
import json
import logging

import gensim
import gensim.downloader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Source: https://www.freecodecamp.org/news/how-to-get-started-with-word2vec-and-then-how-to-make-it-work-d0a2fca9dad3/
# Source: https://code.google.com/archive/p/word2vec/


def text_to_csv():
    datasetSentences = pd.read_csv('data/movies/more/datasetSentences.txt', sep="\t", header=0)
    datasetSplit = pd.read_csv('data/movies/more/datasetSplit.txt', sep=",", header=0)

    data = datasetSentences.merge(datasetSplit, on='sentence_index', how='left')
    data.to_csv('more/more.csv')

    dictionary = pd.read_csv('data/movies/more/dictionary.txt', sep="|", header=None)
    dictionary.columns = ["phrase", "phrase ids"]
    sentiment_labels = pd.read_csv('data/movies/more/sentiment_labels.txt', sep="|", header=0)

    phrases = dictionary.merge(sentiment_labels, on='phrase ids', how='left')
    phrases.to_csv('more/phrases.csv')

    # print(datasetSentences)
    # print(datasetSplit)
    # print(more)


def initialize_vector(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 10000 == 0):
                logging.info("read {0} lines".format(i))
                # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess(line)


def initialize_vector_from_snippets():
    with open('data/movies/more/phrases.csv', 'rb') as f:
        for i, line in enumerate(f):
            if (i % 10000 == 0):
                logging.info("read {0} lines".format(i))
                # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess(line)

def load_sentences_from_csv():
    data = pd.read_csv('data/movies/more/data.csv', header=0, index_col=0)
    return data


def load_phrases_from_csv():
    phrases = pd.read_csv('data/movies/more/phrases.csv', header=0, index_col=0)
    return phrases


def preprocess_text(sentence):
    return sentence.strip().lower()


def split_sentence(sentence):
    words = sentence.split(' ')
    return words


def count_words(words):
    return len(words)


def fill_up_sentence(words, maxlen):
    while len(words) < maxlen:
        words.append('___')
    return words


def convert_sentiment_to_label(sentiment):
    label = 0
    if 0.8 < sentiment <= 1.0:
        label = 4
    elif 0.6 < sentiment <= 0.8:
        label = 3
    elif 0.4 < sentiment <= 0.6:
        label = 2
    elif 0.2 < sentiment <= 0.4:
        label = 1
    elif 0 < sentiment <= 0.2:
        label = 0
    return label

def convert_twitter_label(label):
    if label == -1:
        label = 0
    elif label == 0:
        label = 1
    elif label == 1:
        label = 2

    return label

def download_googlenews_pretrained_model():
    google_news = gensim.downloader.load('word2vec-google-news-300')
    return google_news


def convert_word_to_vector(words, model, dict):
    new = []
    for x in words:
        try:
            new.append(model[x])
        except KeyError:
            new.append(dict[x])
    return new


def convert_words_to_sentence(words):
    sentence = ""
    for word in words:
        sentence += word
        sentence += " "
    sentence.rstrip()
    return sentence


def load_movie_reviews():
    # Load the phrases and sentences more
    phrases = load_phrases_from_csv()
    sentences = load_sentences_from_csv()

    # find out the sentiment values for the sentences
    dataset = sentences.merge(phrases, left_on='sentence', right_on='phrase', how='left')
    dataset = dataset.drop(columns=['phrase', 'phrase ids'])

    # drop rows, where there is no sentiment value
    dataset = dataset[dataset['sentiment values'].notna()]

    # preprocess the sentences
    dataset['sentence'] = dataset['sentence'].apply(preprocess_text)

    # split sentences into words and append to dataframe
    dataset['words'] = dataset['sentence'].apply(split_sentence)

    # convert sentiment values to labels 1-5 and drop rows with label = 0
    dataset['label'] = dataset['sentiment values'].apply(convert_sentiment_to_label)

    # find out max length of sentence and fill up other sentences
    dataset['sentence_length'] = dataset['words'].apply(count_words)
    maxlen = dataset['sentence_length'].max()
    dataset['words'] = dataset['words'].apply(fill_up_sentence, args=(maxlen,))
    dataset['sentence_length'] = dataset['words'].apply(count_words)
    dataset = dataset.drop(columns=['sentence_length'])

    # convert words back to sentences
    dataset['sentence'] = dataset['words'].apply(convert_words_to_sentence)

    word2vec_model = download_googlenews_pretrained_model()

    # build vector dictionary for unknown words and convert words to vectors
    unknown_words = dict()
    for i in range(0, len(dataset.index)):
        for word in dataset.iloc[i, 4]:
            try:
                word2vec_model[word]
            except KeyError:
                if word not in unknown_words:
                    word_vector = np.random.uniform(-0.25, 0.25, 300)
                    unknown_words[word] = word_vector
    dataset['vectors'] = dataset['words'].apply(convert_word_to_vector, args=(word2vec_model, unknown_words,))
    dataset = dataset.drop(columns=['words'])

    # drop unnecessary columns and split dataset in train, test, dev
    dataset = dataset.drop(columns=['sentence_index', 'sentiment values'])
    train = dataset[dataset['splitset_label'] == 1]
    test = dataset[dataset['splitset_label'] == 2]
    dev = dataset[dataset['splitset_label'] == 3]
    train = train.drop(columns=['splitset_label'])
    test = test.drop(columns=['splitset_label'])
    dev = dev.drop(columns=['splitset_label'])

    return train, test, dev, word2vec_model


def load_movie_reviews2():
    # Load the phrases and sentences more
    phrases = load_phrases_from_csv()
    sentences = load_sentences_from_csv()

    # find out the sentiment values for the sentences
    dataset = sentences.merge(phrases, left_on='sentence', right_on='phrase', how='left')
    dataset = dataset.drop(columns=['phrase', 'phrase ids'])

    # drop rows, where there is no sentiment value
    dataset = dataset[dataset['sentiment values'].notna()]

    # convert sentiment values to labels 1-5 and drop rows with label = 0
    dataset['label'] = dataset['sentiment values'].apply(convert_sentiment_to_label)

    # drop unnecessary columns and split dataset in train, test, dev
    dataset = dataset.drop(columns=['sentence_index', 'sentiment values'])
    train = dataset[dataset['splitset_label'] == 1]
    test = dataset[dataset['splitset_label'] == 2]
    dev = dataset[dataset['splitset_label'] == 3]
    train = train.drop(columns=['splitset_label'])
    test = test.drop(columns=['splitset_label'])
    dev = dev.drop(columns=['splitset_label'])

    return train, test, dev

def build_twitter():
    # target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
    df = pd.read_csv('data/twitter/kaggle_twitter.csv', encoding='iso-8859-1')
    df = df.rename(columns={'clean_text': 'sentence', 'category': 'label'})
    df = df.dropna()
    df = shuffle(df, random_state=42)
    df = df.head(25000)

    # preprocess the sentences
    df['sentence'] = df['sentence'].apply(preprocess_text)

    # split sentences into words and append to dataframe
    df['words'] = df['sentence'].apply(split_sentence)

    # find out max length of sentence and fill up other sentences
    df['sentence_length'] = df['sentence'].apply(count_words)
    maxlen = df['sentence_length'].max()
    print(maxlen)
    df['words'] = df['words'].apply(fill_up_sentence, args=(maxlen,))
    df['sentence_length'] = df['words'].apply(count_words)
    df = df.drop(columns=['sentence_length'])

    # convert words back to sentences
    df['sentence'] = df['words'].apply(convert_words_to_sentence)

    df['label'] = df['label'].apply(convert_twitter_label)

    df = df.drop(columns=['words'])

    # Features and Labels
    dev = df.iloc[20000:25000]
    df = df.drop(df.tail(5000).index)
    train = df.sample(frac=0.8, random_state=25)
    test = df.drop(train.index)
    datasets = [train, test, dev]
    build_files_for_torchtext(datasets)

#expects list of dataframes - train, test and dev
def build_files_for_torchtext(datasets):
    i = 0
    for ds in datasets:
        ds.reset_index()
        dics = []
        for index, row in ds.iterrows():
            sentence = row['sentence']
            label = row['label']
            dic = {"sentence": sentence, "label": label}
            dics.append(dic)

        if i == 0:
            split = 'train'
        elif i == 1:
            split = 'test'
        else:
            split = 'dev'

        with open('data/twitter/' + split + '.json', 'w') as fp:
            fp.write('\n'.join(json.dumps(i) for i in dics))

        i += 1
