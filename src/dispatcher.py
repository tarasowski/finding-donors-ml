import sklearn.ensemble as ensemble 


MODELS = {
        'ada_boost': ensemble.AdaBoostClassifier(random_state=7),
        'random_forest': ensemble.RandomForestClassifier(random_state=7),
        'gradient_boosting': ensemble.GradientBoostingClassifier(random_state=7)
        }

GRID_PARAMETERS = {
        'ada_boost': None,
        'random_forest': {
                'n_estimators': [150, 200],
                'max_depth': [100, 110]
                },
        'gradient_boosting': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.1, 0.3, 1]
                }
        }
