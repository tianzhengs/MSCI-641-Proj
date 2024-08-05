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

local= True

def read_csv(fpath):
    with open(fpath) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]



def load():
    data=[]
    with open(r"C:\Tianzheng\OneDrive - University of Waterloo\24 S\MSCI 641\project\task-1\train.jsonl", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    x_train = []
    y_train = []

    for line in data:
        a=re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", line['postText'][0])
        x_train.append(a)
        b=re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", ' '.join(line['targetParagraphs']))
        x_train.append(b)
        y_train.append(line['tags'][0])
        y_train.append(line['tags'][0])


    data=[]
    with open(r"C:\Tianzheng\OneDrive - University of Waterloo\24 S\MSCI 641\project\task-1\val.jsonl", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    x_val = []
    y_val = []

    for line in data:
        a=re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", line['postText'][0])
        x_val.append(a)
        b=re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", ' '.join(line['targetParagraphs']))
        x_val.append(b)
        y_val.append(line['tags'][0])
        y_val.append(line['tags'][0])

    data=[]
    with open(r"C:\Tianzheng\OneDrive - University of Waterloo\24 S\MSCI 641\project\task-1\test.jsonl", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    x_test = []

    for line in data:
        a=re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", line['postText'][0])
        x_test.append(a)
        b=re.sub(r"[!#$%&()*+/:,;.<=>@[\]^`{|}~\t\n\\]", " ", ' '.join(line['targetParagraphs']))
        x_test.append(b)


    return x_train, y_train, x_val, y_val, x_test
def train_and_test(n_gram=(1,1)):
    print(f"Training and testing with n_gram={n_gram}")
    x_train, y_train, x_val, y_val, x_test = load()


    if n_gram == (1,1):
        flag = 'uni'
    elif n_gram == (2,2):
        flag = 'bi'
    else:
        flag = 'uni_bi'


    x_trainval = x_train + x_val
    y_trainval = y_train + y_val
    # when using a validation set, set the test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.
    test_val_flags = [-1] * len(x_train) + [0] * len(x_val)
    ps = PredefinedSplit(test_val_flags)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    # Uni-gram
    param_grid = {
        'vect__ngram_range': [n_gram],
        'tfidf__use_idf': [True, False],
        'tfidf__smooth_idf': [True, False],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=ps, scoring='accuracy')

    grid_search.fit(x_trainval, y_trainval)

    r=grid_search.cv_results_
    print(r)

    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    print("Best score on validation set:")
    print(grid_search.best_score_)


    best_pipeline = grid_search.best_estimator_



    with open(f'./data/w2vec_{flag}.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)

    # preds = best_pipeline.predict(x_test)
    # print(f"Accuracy on test set: {accuracy_score(y_test, preds)}")

def main():
    """Implement your assignment solution here"""
    for n_gram in [(1,1), (2,2), (1,2)]:
        train_and_test(n_gram=n_gram)

if __name__ == "__main__":
    print("Starting main at", time.strftime("%H:%M:%S", time.localtime()))
    main()
    print("Ending main at", time.strftime("%H:%M:%S", time.localtime()))