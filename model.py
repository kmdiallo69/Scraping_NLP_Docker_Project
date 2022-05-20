
""""
Build a classifer
"""
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.feature_extraction.text import  TfidfVectorizer

import pandas as pd

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import pickle

MODEL_PATH = './fastapi/model/'


def build_model():
    """
    :return:
    """
    print('Starting........')
    # load dataset
    df = pd.read_csv('./data/dataset.csv')
    le = LabelEncoder()
    labels = df['label'].to_list()
    # transform labels columns using LabelEncoder (0: no-toxic,1: toxic)
    y = le.fit_transform(labels)
    df.loc[:, 'label'] = y
    # save annotated dataset
    df.to_csv('./data/dataset.csv', index=False)

    print('MODEL 1: TFIDF')
    print('\tCreating model .........')

    # TFIDF (You can implement other model such as LSTM.. Transformer HuggingFace..)
    vectorizer = TfidfVectorizer(ngram_range=(1,4))
    tweets = df['text'].to_list()
    x = vectorizer.fit_transform(tweets)
    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

    kernels = ['linear']
    gammas = [0.1]
    cs = [10]
    param_grid = {'C': cs, 'gamma': gammas, 'kernel': kernels}
    grid = GridSearchCV(SVC(), param_grid=param_grid, refit=True, verbose=3)
    print('\t\tFitting model.......')
    grid.fit(x_train, y_train)
    # best params
    print(grid.best_params_)
    # best estimator
    print(grid.best_estimator_)
    # inference
    predicted = grid.predict(x_test)
    print(classification_report(y_test, predicted))
    # save vectorizer
    vectorizer_filename = 'vectorizer_tfidf.sav'

    # save vectorizer in fastapi folder for further using:
    pickle.dump(vectorizer, open(MODEL_PATH+vectorizer_filename, 'wb'))
    # save model in fastapi folder
    model_filename = 'finalized_tfidf.sav'

    pickle.dump(grid, open(MODEL_PATH + model_filename, 'wb'))

# if __name__ == '__main__':
#     build_model()


