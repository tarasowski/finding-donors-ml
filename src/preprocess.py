import os
import pandas as pd
import numpy as np
from time import time
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

TRAINING_DATA = os.environ.get('TRAINING_DATA')
ENV = os.environ.get('ENV')
MODEL = str(os.environ.get('MODEL'))


df = pd.read_csv(TRAINING_DATA)

n_records = df.shape[0]
n_greater_50k = df.loc[df['income'].str.contains('>50K'), :].shape[0]
n_at_most_50k = df.loc[df['income'].str.contains('<=50'), :].shape[0]
greater_percent = n_greater_50k / n_records * 100

print(f'Total number of records: {format(n_records)}')
print(f'Individuals making more than $50,000: {format(n_greater_50k)}')
print(f'Individuals making at most $50,000: {format(n_at_most_50k)}')
print(f'Percentage of individuals making more than $50,000: {format(greater_percent)}')

features_raw = df.drop('income', axis=1)
income_raw = df['income']

# logarithmic transformation - reduces the range of values caused by outliers
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# normalizing numerical features  
# feature scaling ensures that each feature is treated equally when applying supervised learners.
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

print(features_log_minmax_transform.head(5))

# one-hot encoding
# learning algorithms expect input to be numeric, which requires conversion of categorical variables
features_final = pd.get_dummies(features_log_minmax_transform)
income = income_raw.map({'<=50K': 0, '>50K': 1})
encoded = list(features_final.columns)
print(f'{len(encoded)} total features after one-hot encoding')

# shuffle and split data
X_train, X_test, y_train, y_test = train_test_split(features_final,
        income,
        test_size=0.2,
        random_state=0)

print(f'Training set has {X_train.shape[0]} samples')
print(f'Testing set has {X_test.shape[0]} samples')

if ENV == 'script':
        joblib.dump(X_train, f'models/x_train.pkl')
        joblib.dump(X_test, f'models/x_test.pkl')
        joblib.dump(y_train, f'models/y_train.pkl')
        joblib.dump(y_test, f'models/y_test.pkl')

