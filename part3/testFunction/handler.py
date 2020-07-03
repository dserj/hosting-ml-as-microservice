import json

import nltk
nltk.data.path=['.data']

from nltk.corpus import stopwords
from string import punctuation

stopwords_eng = stopwords.words('english')

def bag_of_words(words):
    bag = {}
    for w in words:
        bag[w] = bag.get(w,0)+1
    return bag

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
# from nltk.util import ngrams
from nltk.util import everygrams

lemmatizer = WordNetLemmatizer()

def extract_features(document):
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w!="" and w not in stopwords_eng and w not in punctuation]
    return [str('_'.join(ngram)) for ngram in list(everygrams(words, max_len=3))]

import pickle
import sys

model_file = open('sa_classifier.pickle', 'rb')
model = pickle.load(model_file)
model_file.close()

from nltk.tokenize import word_tokenize

def get_sentiment(review):
    #words = word_tokenize(review)
    words = extract_features(review)
    words = bag_of_words(words)
    return model.classify(words)

def hello(event, context):

    body = {
        "result": get_sentiment(event['review']),
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response