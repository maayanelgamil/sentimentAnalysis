import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import linear_model
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def pre_process_data(data):
    # remove unnecessary columns
    clean_data_relevant = data[["URLID","Ticker","title_clean","text_clean","classification"]]

    # change classification to numbers
    dic_replace = {"negative": -1, "neutral": 0, "positive": 1}
    clean_data_relevant["classification"] = clean_data_relevant["classification"].replace(dic_replace)

    return clean_data_relevant


# feature of the title length of each article
class TitleFeature(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, titles):
        result = []
        max_title_len = len(max(titles, key=len))
        for title in titles:
            result.append({'length': (len(title) / max_title_len), })
        return result


# feature based on how many matches to positive and negative words
class PosNegWordsInTextFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        ps = PorterStemmer()
        # create set of positive & negative words
        with open('LM_Negative.csv', 'r') as f:
            neg_lines = f.readlines()
            self.negative_set = set([ps.stem(str(line.strip()).lower()) for line in neg_lines])

        with open('LM_Positive.csv', 'r') as f:
            pos_lines = f.readlines()
            self.positive_set = set([ps.stem(str(line.strip()).lower()) for line in pos_lines])

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        result = []
        for text in texts:
            negative_counter = 0
            positive_counter = 0
            word_list = text.split()
            for word in word_list:
                if word in self.negative_set:
                    negative_counter += 1
                elif word in self.positive_set:
                    positive_counter += 1
            result.append(
                {'pos_score': (positive_counter / len(word_list)),
                 'neg_score': (negative_counter / len(word_list))}
            )
        return result


# chooses the relevant column from the dataframe
class dataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.key]




# region define feature_list & feature_weights for pipeline

title_column_name = "title_clean"
article_column_name = "text_clean"

feature_list = [
            # Pipeline for pulling TfidfVectorizer feature from articles body text
            ('article_text_Tfid', Pipeline([
                ('selector', dataSelector(key=article_column_name)),
                ('tfidf', TfidfVectorizer(min_df = 0.01, max_df = 0.5)),
            ])),
            # Pipeline for standard bag-of-words model for articles body
            ('article_text_bag_of_words', Pipeline([
                ('selector', dataSelector(key=article_column_name)),
                ('tfidf', TfidfVectorizer(min_df = 0.01, max_df = 0.5)),
            ])),
            # Pipeline for standard bag-of-words model for title
            ('title_bag_of_words', Pipeline([
                ('selector', dataSelector(key=title_column_name)),
                ('tfidf', TfidfVectorizer(min_df = 0.01, max_df = 0.5)),
            ])),
            # Pipeline for pulling defined features for title
            ('title_features', Pipeline([
                ('selector', dataSelector(key=title_column_name)),
                ('title', TitleFeature()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
            #Pipeline for pulling as feature negative and positive words for article body
            ('pos_neg_words_in_body', Pipeline([
                ('selector', dataSelector(key= article_column_name)),
                ('pos_neg_body', PosNegWordsInTextFeature()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
             #Pipeline for pulling as feature negative and positive words for article title
                ('pos_neg_words_in_title', Pipeline([
                ('selector', dataSelector(key=title_column_name)),
                ('pos_neg_title', PosNegWordsInTextFeature()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
        ]
feature_weights = {
            'title_features': 1,
            'title_bag_of_words': 0.5,
            'article_text_bag_of_words':0.5,
            'article_text_Tfid': 1,
            'pos_neg_words_in_body':10,
            'pos_neg_words_in_title': 10,
        }

# endregion


def classify_by_model(df, model, parameters, with_grid_search, test_set_size):
    pipeline = Pipeline([
        # Use FeatureUnion to combine the different features
        ('union', FeatureUnion(transformer_list=feature_list, transformer_weights=feature_weights,
                               )),
        # Use a clf classifier on the combined features
        ('clf', model),
    ])
    if with_grid_search:
        clf = GridSearchCV(pipeline, parameters)
    else:
        clf = pipeline

    train, test = train_test_split(df, test_size=test_set_size, random_state=21)
    print(" ***** Train/Test sets size ***** ")
    print("Train size: " + str(len(train)))
    print("Test size: " + str(len(test)))

    train_data = train
    train_target = train.classification

    test_data = test
    test_target = test.classification

    clf = clf.fit(train_data, train_target)
    if with_grid_search:
        print("***** GridSearchCV Results *****")
        print('Best score: ', clf.best_score_)
        print('Best params: ', clf.best_params_)
    test_target_pred = clf.predict(test_data)

    score = metrics.accuracy_score(test_target, test_target_pred)

    print("***** Accuracy score *****")
    print(score)


def extract_features(data):

    clean_df = pre_process_data(data)

    my_model = linear_model.SGDClassifier()

    # parameters for gridSearch in order to find the optimal parameters for the classifiers
    my_parameters = {'union__title_bag_of_words__tfidf__max_df': (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 1.0),
                     # 'union__title_bag_of_words__tfidf__ngram_range': ((1, 1), (1, 2)), # unigrams or bigrams
                     # 'clf__alpha': (0.0001, 0.01,1.0),
                     # 'clf__loss':['hinge','epsilon_insensitive','modified_huber'],
                     # 'clf__penalty': ('l2', 'elasticnet'),
                     # 'clf__average':[False, True]
                     }
    with_grid_search = True

    test_set_size = 0.15

    classify_by_model(clean_df, my_model, my_parameters, with_grid_search, test_set_size)

