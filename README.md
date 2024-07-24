# Sentiment-Analysis

This repository contains code and resources for performing sentiment analysis using various models and techniques.
It focuses on analyzing text data to determine the sentiment expressed, whether positive, negative, or neutral.

I developed this project during my bachelor's studies for two different courses.

In the first part I reimplemented the [C-LSTM Paper](https://arxiv.org/pdf/1511.08630) by myself.

In the second part I compared the results with textblob and spaCy.

## Data

To evaluate these datasets I have used:

* A twitter dataset from Kaggle with three different sentiment labels
* Movie reviews from Rotten Tomatoes with five different sentiment labels (which were used in [this paper](https://aclanthology.org/D13-1170/))

## Results

||C-LSTM|textblob|spaCy|
|--|--|--|--|
|movies|	39,35%|	29,16%	|41,09%|
|twitter	|64,95%|	54,20%|	93,30%|

