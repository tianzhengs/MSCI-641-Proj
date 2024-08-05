import json
import os
import re
import sys
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import time
from text_preprocessing import preprocess_text
import pandas as pd

# import gensim's Word2Vec module
from gensim.models import Word2Vec



import nltk
nltk.download('punkt') # dependency of sent_tokenize function


data=[]
with open(r"C:\Tianzheng\OneDrive - University of Waterloo\24 S\MSCI 641\project\task-1\train.jsonl", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

all_sentences=[]
for line in data:
    sentences=nltk.sent_tokenize(line['postText'][0])
    sentences = [re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", s.lower()) for s in sentences]
    sentences = [line.strip().split() for line in sentences]
    all_sentences.extend(sentences)

    sentences=nltk.sent_tokenize(' '.join(line['targetParagraphs']))
    sentences = [re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", s.lower()) for s in sentences]
    sentences = [line.strip().split() for line in sentences]
    all_sentences.extend(sentences)

    w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    w2v.save('./cus_w2v.model')
