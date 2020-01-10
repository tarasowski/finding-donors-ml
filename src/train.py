import os
import pandas as pd
import numpy as np
from time import time
import joblib
from . import visuals as vs
from . import dispatcher
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL = str(os.environ.get('MODEL'))
X_TRAIN = str(os.environ.get('X_TRAIN')) 
X_TEST = str(os.environ.get('X_TEST')) 
Y_TRAIN = str(os.environ.get('Y_TRAIN')) 
Y_TEST = str(os.environ.get('Y_TEST')) 

X_train = joblib.load(os.path.join(f'{X_TRAIN}')).values
X_test = joblib.load(os.path.join(f'{X_TEST}')).values
y_train = joblib.load(os.path.join(f'{Y_TRAIN}')).values
y_test = joblib.load(os.path.join(f'{Y_TEST}')).values

print(f'Loaded -> Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}')

clf = dispatcher.MODELS[MODEL]
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(f'Accuracy score on testing data: {accuracy_score(y_test, predictions)}')
print(f'F-score on testing data: {fbeta_score(y_test, predictions, beta=0.5)}')

joblib.dump(clf, f'models/{MODEL}.pkl')
