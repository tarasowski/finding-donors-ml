import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from . import dispatcher
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

clf = joblib.load(os.path.join('models', f'{MODEL}.pkl'))

parameters = dispatcher.GRID_PARAMETERS[MODEL] or None
scorer = make_scorer(fbeta_score, beta=0.5)

grid_obj = GridSearchCV(clf, parameters, scoring=scorer, verbose=10)
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_

predictions = clf.fit(X_train, y_train).predict(X_test)
best_predictions = best_clf.predict(X_test)

result_acc = accuracy_score(y_test, best_predictions) - accuracy_score(y_test, predictions)  
result_fbeta = fbeta_score(y_test, best_predictions, beta=0.5) - fbeta_score(y_test, predictions, beta=0.5)

print('Unoptimized model\n----')
print(f'Accuracy score on testing data: {accuracy_score(y_test, predictions)}')
print(f'F-score on testing data: {fbeta_score(y_test, predictions, beta=0.5)}')
print('\nOptimized Model\n----')
print(f'Final accuracy score on the testing data: {accuracy_score(y_test, best_predictions)}')
print(f'Final F-score on the testing data: {fbeta_score(y_test, best_predictions, beta=0.5)}')

def save_model(acc, fbeta):
    if acc >= 0 and fbeta > 0:
        print(f'Saving {MODEL}')
        joblib.dump(best_clf, f'models/{MODEL}.pkl')
    else:
        print('No model improvements made.')

save_model(result_acc, result_fbeta)
