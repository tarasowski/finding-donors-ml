import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL = str(os.environ.get('MODEL'))

# adjust the script to predict on real data
# dummy prediction on test data
X_PREDICTION = str(os.environ.get('X_TEST')) 
Y_PREDICTION = str(os.environ.get('Y_TEST')) 

X_test = joblib.load(os.path.join(f'{X_PREDICTION}')).values
y_test = joblib.load(os.path.join(f'{Y_PREDICTION}')).values

clf = joblib.load(os.path.join('models', f'{MODEL}.pkl'))
predictions = clf.predict(X_test)

print(f'Accuracy score on testing data (PREDICTION): {accuracy_score(y_test, predictions)}')
print(f'F-score on testing data (PREDICTION): {fbeta_score(y_test, predictions, beta=0.5)}')
